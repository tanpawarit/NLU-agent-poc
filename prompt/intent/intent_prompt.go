package intent

import (
	"context"
	_ "embed"
	"fmt"
	"strings"

	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/schema"
)

type intentModelConfig struct {
	IntentList string `envconfig:"NLU_INTENT" default:"greet:0.1, purchase_intent:0.8, inquiry_intent:0.7, support_intent:0.6, complain_intent:0.6, complaint:0.5, cancel_order:0.4, ask_price:0.6, compare_product:0.5, delivery_issue:0.7"`
}

//go:embed intent_template.txt
var intentSystemTemplate string

// RenderintentSystem renders the intent system prompt via Eino prompt component.
// This triggers Prompt callbacks and returns the final system prompt string.
func RenderintentSystem(ctx context.Context, intentConfig *intentModelConfig) (string, error) {
	if intentConfig == nil {
		return "", fmt.Errorf("intent config is nil")
	}

	// Safely render known tokens only to avoid interfering with JSON braces in template
	content := strings.NewReplacer(
		"{TD}", "<||>",
		"{RD}", "##",
		"{CD}", "<|COMPLETE|>",
		"{intent_list}", intentConfig.IntentList,
	).Replace(intentSystemTemplate)

	// Wrap via Eino prompt component using a messages placeholder to emit callbacks
	tpl := prompt.FromMessages(
		schema.FString,
		schema.MessagesPlaceholder("system_messages", false),
	)
	msgs, err := tpl.Format(ctx, map[string]any{
		"system_messages": []*schema.Message{schema.SystemMessage(content)},
	})
	if err != nil {
		return "", fmt.Errorf("intent prompt callbacks: %w", err)
	}
	if len(msgs) == 0 || msgs[0] == nil {
		return "", fmt.Errorf("intent prompt callbacks: empty result")
	}
	return msgs[0].Content, nil
}

type IntentResult struct {
	Name       string         `json:"name"`
	Confidence float64        `json:"confidence"`
	Priority   float64        `json:"priority_score"`
	Meta       map[string]any `json:"meta"` // {"source":"config", "closest_match":true?}
}

type LanguageResult struct {
	Code        string         `json:"code"` // ISO 639-3
	Confidence  float64        `json:"confidence"`
	PrimaryFlag int            `json:"primary_flag"`
	Meta        map[string]any `json:"meta"` // {"script":"thai","detected_tokens":4}
}

type IntentOutput struct {
	Intents   []IntentResult   `json:"intents"`
	Languages []LanguageResult `json:"languages"`
}
