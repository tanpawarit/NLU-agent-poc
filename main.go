package main

import (
	"context"
	"fmt"
	"os"

	"github.com/cloudwego/eino-ext/components/model/gemini"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
	"github.com/joho/godotenv"
	"google.golang.org/genai"
)

func main() {
	_ = godotenv.Load()

	// systemPrompt, err := loadSystemPrompt()
	// if err != nil {
	// 	fmt.Println("failed to load system prompt:", err)
	// 	return
	// }

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("missing GEMINI_API_KEY")
		return
	}
	baseURL := os.Getenv("GEMINI_BASE_URL")
	modelName := "gemini-2.5-flash"
	temperature := float32(0.4)
	maxTokens := 2024
	thinkingBudget := int32(2000)
	ctx := context.Background()

	clientCfg := &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	}
	if baseURL != "" {
		clientCfg.HTTPOptions.BaseURL = baseURL
	}

	client, err := genai.NewClient(ctx, clientCfg)
	if err != nil {
		fmt.Println("failed to create Gemini client:", err)
		return
	}

	chatModel, err := gemini.NewChatModel(ctx, &gemini.Config{
		Client:      client,
		Model:       modelName,
		Temperature: &temperature,
		MaxTokens:   &maxTokens,
		ThinkingConfig: &genai.ThinkingConfig{
			IncludeThoughts: false,
			ThinkingBudget:  &thinkingBudget,
		},
	})
	if err != nil {
		fmt.Println("failed to create chat model:", err)
		return
	}

	chatGraph := compose.NewGraph[[]*schema.Message, *schema.Message]()
	if err := chatGraph.AddChatModelNode("llm", chatModel); err != nil {
		fmt.Println("failed to add chat model node:", err)
		return
	}
	if err := chatGraph.AddEdge(compose.START, "llm"); err != nil {
		fmt.Println("failed to link start to llm:", err)
		return
	}
	if err := chatGraph.AddEdge("llm", compose.END); err != nil {
		fmt.Println("failed to link llm to end:", err)
		return
	}

	chatRunnable, err := chatGraph.Compile(ctx)
	if err != nil {
		fmt.Println("failed to compile graph:", err)
		return
	}

}
