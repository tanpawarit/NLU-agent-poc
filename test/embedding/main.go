package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	geminiembed "github.com/cloudwego/eino-ext/components/embedding/gemini"
	"github.com/joho/godotenv"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"google.golang.org/genai"
)

// Basic configuration for the Milvus collection, embedding model, and dataset
const (
	// Target collection to (re)create and write into
	collectionName = "articles"
	// Max lengths for string fields in Milvus VarChar columns
	titleMaxLength       = 1024
	linkMaxLength        = 1024
	publicationMaxLength = 512
	// Name of the float vector field that stores title embeddings
	titleVectorField = "title_vector"
	// Name of the default index created on the vector field
	defaultVectorIndexName = "title_vector_idx"
	// Gemini embedding model to use when generating vectors
	defaultEmbeddingModel = "gemini-embedding-001"
	// Source dataset containing article metadata (titles, links, etc.)
	datasetPath = "data/medium_articles_2020_dpr_a13e0377ae.json"
)

// ensureArticlesCollection drops an existing collection (if any) and recreates
// it with the expected schema and a cosine similarity index for the vector field.
// dim must match the dimension produced by the embedding model.
func ensureArticlesCollection(ctx context.Context, cli *milvusclient.Client, dim int) error {
	if dim <= 0 {
		return fmt.Errorf("invalid embedding dimension: %d", dim)
	}

	// Check whether the target collection already exists.
	// For this demo, we drop it if present to recreate a clean schema/index.
	// NOTE: In production, avoid dropping live collections; prefer migrations or conditional create.
	// Check and drop the collection for a clean setup (demo-only behavior)
	hasCollection, err := cli.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		return err
	}
	if hasCollection {
		if err := cli.DropCollection(ctx, milvusclient.NewDropCollectionOption(collectionName)); err != nil {
			return err
		}
	}

	// Define schema: primary key, vector field, and supporting metadata fields
	schema := entity.NewSchema().
		WithName(collectionName).
		WithDynamicFieldEnabled(true).
		WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(false)).
		WithField(entity.NewField().WithName(titleVectorField).WithDataType(entity.FieldTypeFloatVector).WithDim(int64(dim))).
		WithField(entity.NewField().WithName("title").WithDataType(entity.FieldTypeVarChar).WithMaxLength(titleMaxLength)).
		WithField(entity.NewField().WithName("link").WithDataType(entity.FieldTypeVarChar).WithMaxLength(linkMaxLength)).
		WithField(entity.NewField().WithName("publication").WithDataType(entity.FieldTypeVarChar).WithMaxLength(publicationMaxLength)).
		WithField(entity.NewField().WithName("reading_time").WithDataType(entity.FieldTypeInt32)).
		WithField(entity.NewField().WithName("claps").WithDataType(entity.FieldTypeInt32)).
		WithField(entity.NewField().WithName("responses").WithDataType(entity.FieldTypeInt32))

	// Create an auto index for the vector field using cosine similarity
	createOption := milvusclient.NewCreateCollectionOption(collectionName, schema).
		WithIndexOptions(
			milvusclient.NewCreateIndexOption(collectionName,
				titleVectorField,
				index.NewAutoIndex(entity.COSINE)).
				WithIndexName(defaultVectorIndexName),
		).WithConsistencyLevel(entity.ClSession)

	return cli.CreateCollection(ctx, createOption)
}

func main() {
	_ = godotenv.Load()

	ctx := context.Background()
	// Load dataset
	type Row struct {
		ID          int64  `json:"id"`
		Title       string `json:"title"`
		Link        string `json:"link"`
		ReadingTime int32  `json:"reading_time"`
		Publication string `json:"publication"`
		Claps       int32  `json:"claps"`
		Responses   int32  `json:"responses"`
	}
	var payload struct {
		Rows []Row `json:"rows"`
	}

	// Open and decode the JSON dataset
	f, err := os.Open(filepath.Clean(datasetPath))
	if err != nil {
		log.Fatalf("open dataset: %v", err)
	}
	defer f.Close()
	if err := json.NewDecoder(f).Decode(&payload); err != nil {
		log.Fatalf("decode dataset: %v", err)
	}
	if len(payload.Rows) == 0 {
		log.Fatal("dataset has no rows")
	}
	log.Printf("dataset rows: %d", len(payload.Rows))

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("missing GEMINI_API_KEY")
	}

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

	// Connect to Milvus
	milvusAddr := os.Getenv("MILVUS_ADDR")
	if milvusAddr == "" {
		log.Fatal("missing MILVUS_ADDR")
	}

	milvus, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address:  milvusAddr,
		Username: os.Getenv("MILVUS_USERNAME"),
		Password: os.Getenv("MILVUS_PASSWORD"),
	})
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if closeErr := milvus.Close(ctx); closeErr != nil {
			log.Printf("failed to close milvus client: %v", closeErr)
		}
	}()

	// Upsert in batches using dataset content.
	// Gemini Embed API limits batch size to <=100 items per request.
	const batchSize = 100
	var vectorDim int
	for start := 0; start < len(payload.Rows); start += batchSize {
		end := start + batchSize
		if end > len(payload.Rows) {
			end = len(payload.Rows)
		}
		batch := payload.Rows[start:end]

		// Prepare columnar slices for the current batch
		ids := make([]int64, len(batch))
		titles := make([]string, len(batch))
		links := make([]string, len(batch))
		publications := make([]string, len(batch))
		readingTimes := make([]int32, len(batch))
		claps := make([]int32, len(batch))
		responses := make([]int32, len(batch))

		for i, r := range batch {
			ids[i] = r.ID
			titles[i] = r.Title
			links[i] = r.Link
			publications[i] = r.Publication
			readingTimes[i] = r.ReadingTime
			claps[i] = r.Claps
			responses[i] = r.Responses
		}

		embeddings, err := embedder.EmbedStrings(ctx, titles)
		if err != nil {
			log.Fatalf("embed batch %d-%d: %v", start, end, err)
		}
		if len(embeddings) != len(batch) {
			log.Fatalf("embed batch %d-%d: got %d embeddings for %d titles", start, end, len(embeddings), len(batch))
		}

		if vectorDim == 0 {
			if len(embeddings[0]) == 0 {
				log.Fatal("gemini returned empty embedding")
			}
			vectorDim = len(embeddings[0])
			log.Printf("gemini embedding dim: %d", vectorDim)
			if err := ensureArticlesCollection(ctx, milvus, vectorDim); err != nil {
				log.Fatal(err)
			}
		}

		vectors := make([][]float32, len(batch))
		for i, emb := range embeddings {
			if len(emb) != vectorDim {
				log.Fatalf("embed batch %d-%d: embedding dim mismatch at index %d (expected %d, got %d)", start, end, i, vectorDim, len(emb))
			}
			vec := make([]float32, vectorDim)
			for j, v := range emb {
				vec[j] = float32(v)
			}
			vectors[i] = vec
		}

		// Build an upsert option in column-based mode and send it
		upsertOption := milvusclient.NewColumnBasedInsertOption(collectionName).
			WithInt64Column("id", ids).
			WithFloatVectorColumn(titleVectorField, vectorDim, vectors).
			WithVarcharColumn("title", titles).
			WithVarcharColumn("link", links).
			WithInt32Column("reading_time", readingTimes).
			WithVarcharColumn("publication", publications).
			WithInt32Column("claps", claps).
			WithInt32Column("responses", responses)

		if _, err = milvus.Upsert(ctx, upsertOption); err != nil {
			log.Fatalf("upsert batch %d-%d: %v", start, end, err)
		}
	}

	// Ensure all data is persisted before exit
	flushTask, err := milvus.Flush(ctx, milvusclient.NewFlushOption(collectionName))
	if err != nil {
		log.Fatal(err)
	}
	if err := flushTask.Await(ctx); err != nil {
		log.Fatal(err)
	}
}
