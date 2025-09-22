package main

import (
	"context"
	"log"
	"os"

	geminiembed "github.com/cloudwego/eino-ext/components/embedding/gemini"
	"github.com/joho/godotenv"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"google.golang.org/genai"
)

func main() {
	_ = godotenv.Load()
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("missing GEMINI_API_KEY")
	}

	// Get the environment variables
	addr := os.Getenv("MILVUS_ADDR")
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

	emb, err := geminiembed.NewEmbedder(ctx, &geminiembed.EmbeddingConfig{
		Client: genaiClient,
		Model:  "gemini-embedding-001",
	})
	if err != nil {
		log.Fatal(err)
	}
	_ = emb

	// Create a client
	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address:  addr,
		Username: username,
		Password: password,
	})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
		return
	}
	defer func() {
		if closeErr := cli.Close(ctx); closeErr != nil {
			log.Printf("failed to close milvus client: %v", closeErr)
		}
	}()

}
