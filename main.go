package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	_ "embed"

	"github.com/cloudwego/eino-ext/components/model/gemini"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
	"github.com/joho/godotenv"
	"google.golang.org/genai"
)

//go:embed system_prompt.txt
var embeddedSystemPrompt string

func loadSystemPrompt() (string, error) {
	if promptPath := os.Getenv("SYSTEM_PROMPT_PATH"); promptPath != "" {
		data, err := os.ReadFile(filepath.Clean(promptPath))
		if err != nil {
			return "", err
		}
		prompt := strings.TrimSpace(string(data))
		if prompt == "" {
			return "", fmt.Errorf("system prompt file %s is empty", promptPath)
		}
		return prompt, nil
	}

	prompt := strings.TrimSpace(embeddedSystemPrompt)
	if prompt == "" {
		return "", fmt.Errorf("embedded system prompt is empty")
	}
	return prompt, nil
}

func main() {
	_ = godotenv.Load()

	systemPrompt, err := loadSystemPrompt()
	if err != nil {
		fmt.Println("failed to load system prompt:", err)
		return
	}

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("missing GEMINI_API_KEY")
		return
	}
	baseURL := os.Getenv("GEMINI_BASE_URL")
	modelName := "gemini-2.5-flash"
	temperature := float32(0.4)
	maxTokens := 1024
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
			IncludeThoughts: true,
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

	history := []*schema.Message{schema.SystemMessage(systemPrompt)}

	reader := bufio.NewReader(os.Stdin)
	fmt.Println("Eino PoC Chatbot (พิมพ์ 'exit' เพื่อออก)")
	for {
		fmt.Print("You: ")
		line, readErr := reader.ReadString('\n')
		if readErr != nil {
			fmt.Println("read error:", readErr)
			return
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if line == "exit" {
			break
		}

		userMsg := schema.UserMessage(line)
		history = append(history, userMsg)

		input := make([]*schema.Message, len(history))
		copy(input, history)

		out, invokeErr := chatRunnable.Invoke(ctx, input)
		if invokeErr != nil {
			fmt.Println("Error:", invokeErr)
			history = history[:len(history)-1]
			continue
		}

		history = append(history, out)
		fmt.Println("Bot:", strings.TrimSpace(out.Content))
	}
}
