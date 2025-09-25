package tools

import (
	"context"
	"fmt"
	"os"
	"strings"

	geminiembed "github.com/cloudwego/eino-ext/components/embedding/gemini"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/cloudwego/eino/schema"
	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"google.golang.org/genai"
)

const (
	defaultArticleTopK     = 5
	articlesCollectionName = "articles"
	titleVectorField       = "title_vector"
	defaultEmbeddingModel  = "gemini-embedding-001"
)

// SearchArticlesInput contains the query and optional parameters for article search.
type SearchArticlesInput struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k,omitempty"`
}

// ArticleSearchResult represents a single article hit returned from Milvus.
type ArticleSearchResult struct {
	ID          string  `json:"id"`
	Title       string  `json:"title"`
	Link        string  `json:"link"`
	Publication string  `json:"publication"`
	ReadingTime int     `json:"reading_time"`
	Claps       int     `json:"claps"`
	Responses   int     `json:"responses"`
	Score       float64 `json:"score"`
}

// SearchArticlesOutput wraps the list of retrieved articles.
type SearchArticlesOutput struct {
	Articles []ArticleSearchResult `json:"articles"`
	Total    int                   `json:"total"`
}

func createSearchArticleTool() tool.BaseTool {
	return utils.NewTool(
		&schema.ToolInfo{
			Name: ToolSearchArticles,
			Desc: "Semantic search over the articles vector database. Provide a natural language question to retrieve relevant Medium-style articles (title, publication, link, engagement metrics).",
			ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
				"query": {
					Type:     "string",
					Desc:     "Free-form question or keywords describing the article you are looking for.",
					Required: true,
				},
				"top_k": {
					Type: "number",
					Desc: "Maximum number of articles to return (default: 5, max: 20).",
				},
			}),
		},
		func(ctx context.Context, in *SearchArticlesInput) (*SearchArticlesOutput, error) {
			if in.Query == "" {
				return nil, fmt.Errorf("query is required")
			}

			topK := in.TopK
			if topK <= 0 {
				topK = defaultArticleTopK
			}
			if topK > 20 {
				topK = 20
			}

			apiKey := strings.TrimSpace(os.Getenv("GEMINI_API_KEY"))
			if apiKey == "" {
				return nil, fmt.Errorf("missing GEMINI_API_KEY")
			}

			genaiClient, err := genai.NewClient(ctx, &genai.ClientConfig{
				APIKey:  apiKey,
				Backend: genai.BackendGeminiAPI,
			})
			if err != nil {
				return nil, fmt.Errorf("create genai client: %w", err)
			}

			embedder, err := geminiembed.NewEmbedder(ctx, &geminiembed.EmbeddingConfig{
				Client: genaiClient,
				Model:  defaultEmbeddingModel,
			})
			if err != nil {
				return nil, fmt.Errorf("create embedder: %w", err)
			}

			embeddings, err := embedder.EmbedStrings(ctx, []string{in.Query})
			if err != nil {
				return nil, fmt.Errorf("embed query: %w", err)
			}
			if len(embeddings) == 0 || len(embeddings[0]) == 0 {
				return nil, fmt.Errorf("embed query: empty embedding returned")
			}

			queryVector := make([]float32, len(embeddings[0]))
			for i, v := range embeddings[0] {
				queryVector[i] = float32(v)
			}

			addr := strings.TrimSpace(os.Getenv("MILVUS_ADDR"))
			if addr == "" {
				return nil, fmt.Errorf("missing MILVUS_ADDR")
			}

			milvusClient, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
				Address:  addr,
				Username: strings.TrimSpace(os.Getenv("MILVUS_USERNAME")),
				Password: strings.TrimSpace(os.Getenv("MILVUS_PASSWORD")),
			})
			if err != nil {
				return nil, fmt.Errorf("create milvus client: %w", err)
			}
			defer milvusClient.Close(ctx)

			loadTask, err := milvusClient.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(articlesCollectionName))
			if err != nil {
				return nil, fmt.Errorf("load collection %s: %w", articlesCollectionName, err)
			}
			if err := loadTask.Await(ctx); err != nil {
				return nil, fmt.Errorf("await collection load: %w", err)
			}

			searchOpt := milvusclient.NewSearchOption(articlesCollectionName, topK, []entity.Vector{entity.FloatVector(queryVector)}).
				WithANNSField(titleVectorField).
				WithOutputFields("title", "link", "publication", "reading_time", "claps", "responses").
				WithSearchParam("metric_type", string(entity.COSINE)).
				WithSearchParam("params", "{\"nprobe\": 10}")

			resultSets, err := milvusClient.Search(ctx, searchOpt)
			if err != nil {
				return nil, fmt.Errorf("search collection: %w", err)
			}

			if len(resultSets) == 0 || resultSets[0].ResultCount == 0 {
				return &SearchArticlesOutput{Articles: nil, Total: 0}, nil
			}

			rs := resultSets[0]
			titleCol := rs.GetColumn("title")
			linkCol := rs.GetColumn("link")
			publicationCol := rs.GetColumn("publication")
			readingTimeCol := rs.GetColumn("reading_time")
			clapsCol := rs.GetColumn("claps")
			responsesCol := rs.GetColumn("responses")

			articles := make([]ArticleSearchResult, 0, rs.ResultCount)
			for idx := 0; idx < rs.ResultCount; idx++ {
				idVal, err := rs.IDs.Get(idx)
				if err != nil {
					return nil, fmt.Errorf("result %d: get id: %w", idx, err)
				}

				title, err := valueAsString(titleCol, idx)
				if err != nil {
					return nil, fmt.Errorf("result %d: decode title: %w", idx, err)
				}

				link, err := valueAsString(linkCol, idx)
				if err != nil {
					return nil, fmt.Errorf("result %d: decode link: %w", idx, err)
				}

				publication, err := valueAsString(publicationCol, idx)
				if err != nil {
					return nil, fmt.Errorf("result %d: decode publication: %w", idx, err)
				}

				readingTime, err := valueAsInt(readingTimeCol, idx)
				if err != nil {
					return nil, fmt.Errorf("result %d: decode reading_time: %w", idx, err)
				}

				claps, err := valueAsInt(clapsCol, idx)
				if err != nil {
					return nil, fmt.Errorf("result %d: decode claps: %w", idx, err)
				}

				responses, err := valueAsInt(responsesCol, idx)
				if err != nil {
					return nil, fmt.Errorf("result %d: decode responses: %w", idx, err)
				}

				articles = append(articles, ArticleSearchResult{
					ID:          fmt.Sprint(idVal),
					Title:       title,
					Link:        link,
					Publication: publication,
					ReadingTime: readingTime,
					Claps:       claps,
					Responses:   responses,
					Score:       float64(rs.Scores[idx]),
				})
			}

			return &SearchArticlesOutput{
				Articles: articles,
				Total:    len(articles),
			}, nil
		},
	)
}

func valueAsString(col column.Column, idx int) (string, error) {
	if col == nil {
		return "", nil
	}
	val, err := col.Get(idx)
	if err != nil {
		return "", err
	}
	s, ok := val.(string)
	if !ok {
		return fmt.Sprintf("%v", val), nil
	}
	return s, nil
}

func valueAsInt(col column.Column, idx int) (int, error) {
	if col == nil {
		return 0, nil
	}
	val, err := col.Get(idx)
	if err != nil {
		return 0, err
	}
	switch v := val.(type) {
	case int:
		return v, nil
	case int32:
		return int(v), nil
	case int64:
		return int(v), nil
	default:
		return 0, fmt.Errorf("unexpected type %T", val)
	}
}
