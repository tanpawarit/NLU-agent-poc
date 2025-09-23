package main

import (
	"context"
	"fmt"
	"log"
	"os"

	geminiembed "github.com/cloudwego/eino-ext/components/embedding/gemini"
	"github.com/joho/godotenv"
	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"google.golang.org/genai"
)

const (
	collectionName        = "articles"
	vectorField           = "title_vector"
	defaultEmbeddingModel = "gemini-embedding-001"
	defaultTopK           = 5
	defaultQueryText      = "How do I use NLP with Python?" // Example query
)

func main() {
	_ = godotenv.Load()
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("missing GEMINI_API_KEY")
	}

	addr := os.Getenv("MILVUS_ADDR")
	if addr == "" {
		log.Fatal("missing MILVUS_ADDR")
	}
	username := os.Getenv("MILVUS_USERNAME")
	password := os.Getenv("MILVUS_PASSWORD")

	ctx := context.Background()

	genaiClient, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		log.Fatal(err)
	}

	embedder, err := geminiembed.NewEmbedder(ctx, &geminiembed.EmbeddingConfig{
		Client: genaiClient,
		Model:  defaultEmbeddingModel,
	})
	if err != nil {
		log.Fatal(err)
	}

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address:  addr,
		Username: username,
		Password: password,
	})
	if err != nil {
		log.Fatalf("create milvus client: %v", err)
	}
	defer func() {
		if closeErr := cli.Close(ctx); closeErr != nil {
			log.Printf("failed to close milvus client: %v", closeErr)
		}
	}()

	// Ensure the collection is loaded for search
	loadTask, err := cli.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(collectionName))
	if err != nil {
		log.Fatalf("load collection %s: %v", collectionName, err)
	}
	if err := loadTask.Await(ctx); err != nil {
		log.Fatalf("await collection load: %v", err)
	}

	queryText := defaultQueryText
	log.Printf("using query: %q", queryText)

	embeddings, err := embedder.EmbedStrings(ctx, []string{queryText})
	if err != nil {
		log.Fatalf("embed query: %v", err)
	}
	if len(embeddings) == 0 || len(embeddings[0]) == 0 {
		log.Fatal("embed query: empty embedding returned")
	}

	queryVector := make([]float32, len(embeddings[0]))
	for i, v := range embeddings[0] {
		queryVector[i] = float32(v)
	}

	searchOpt := milvusclient.NewSearchOption(collectionName, defaultTopK, []entity.Vector{entity.FloatVector(queryVector)}).
		WithANNSField(vectorField).
		WithOutputFields("title", "link", "publication", "reading_time", "claps", "responses").
		WithSearchParam("metric_type", string(entity.COSINE)).
		WithSearchParam("params", "{\"nprobe\": 10}")

	resultSets, err := cli.Search(ctx, searchOpt)
	if err != nil {
		log.Fatalf("search collection: %v", err)
	}

	if len(resultSets) == 0 {
		log.Println("no results returned from search")
		return
	}

	for queryIdx, rs := range resultSets {
		if rs.ResultCount == 0 {
			log.Printf("query %d: no hits", queryIdx)
			continue
		}

		titleCol := rs.GetColumn("title")
		linkCol := rs.GetColumn("link")
		publicationCol := rs.GetColumn("publication")
		readingTimeCol := rs.GetColumn("reading_time")
		clapsCol := rs.GetColumn("claps")
		responsesCol := rs.GetColumn("responses")

		for idx := 0; idx < rs.ResultCount; idx++ {
			idVal, err := rs.IDs.Get(idx)
			if err != nil {
				log.Printf("query %d result %d: get id: %v", queryIdx, idx, err)
				continue
			}

			title, err := valueAsString(titleCol, idx)
			if err != nil {
				log.Printf("query %d result %d: title decode: %v", queryIdx, idx, err)
			}

			link, err := valueAsString(linkCol, idx)
			if err != nil {
				log.Printf("query %d result %d: link decode: %v", queryIdx, idx, err)
			}

			publication, err := valueAsString(publicationCol, idx)
			if err != nil {
				log.Printf("query %d result %d: publication decode: %v", queryIdx, idx, err)
			}

			readingTime, err := valueAsInt(readingTimeCol, idx)
			if err != nil {
				log.Printf("query %d result %d: reading_time decode: %v", queryIdx, idx, err)
			}

			claps, err := valueAsInt(clapsCol, idx)
			if err != nil {
				log.Printf("query %d result %d: claps decode: %v", queryIdx, idx, err)
			}

			responses, err := valueAsInt(responsesCol, idx)
			if err != nil {
				log.Printf("query %d result %d: responses decode: %v", queryIdx, idx, err)
			}

			score := float64(rs.Scores[idx])

			fmt.Printf("#%d id=%v score=%.4f\n", idx+1, idVal, score)
			fmt.Printf("  title: %s\n", title)
			fmt.Printf("  publication: %s | reading_time=%dmin | claps=%d | responses=%d\n", publication, readingTime, claps, responses)
			fmt.Printf("  link: %s\n\n", link)
		}
	}

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
	case int32:
		return int(v), nil
	case int64:
		return int(v), nil
	case int:
		return v, nil
	default:
		return 0, fmt.Errorf("unexpected type %T", val)
	}
}
