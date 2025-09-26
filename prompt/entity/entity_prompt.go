package intent

import (
	"context"
	_ "embed"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/schema"
)

type entityModelConfig struct {
	Entities string `envconfig:"NLU_ENTITY" default:"product,quantity,brand,price,color,model,spec,budget,warranty,delivery"`
}

type entityModelInput struct {
	IntentName      string
	RequiredKeys    []string
	AllowedEntities []string
	UserMessage     string
	Language        string
}

//go:embed entity_template.txt
var entitySystemTemplate string

// RenderEntitySystem renders the entity system prompt via Eino prompt component.
func RenderEntitySystem(ctx context.Context, in *entityModelInput) (string, error) {
	if in == nil {
		return "", fmt.Errorf("entity input is nil")
	}
	// Normalize allowed entities and ensure uniqueness
	allEntities := append([]string{}, in.AllowedEntities...)
	uniq := map[string]struct{}{}
	for _, e := range allEntities {
		uniq[strings.TrimSpace(e)] = struct{}{}
	}

	// allowed_entities_csv
	var allowed []string
	for k := range uniq {
		allowed = append(allowed, k)
	}
	allowedCSV := strings.Join(allowed, ",")

	// Required เป็น CSV
	reqCSV := strings.Join(in.RequiredKeys, ",")

	content := strings.NewReplacer(
		"{{intent_name}}", in.IntentName,
		"{{required_keys_csv}}", reqCSV,
		"{{allowed_entities_csv}}", allowedCSV,
		"{{user_message}}", in.UserMessage,
		"{{language}}", in.Language,
	).Replace(entitySystemTemplate)

	tpl := prompt.FromMessages(
		schema.FString,
		schema.MessagesPlaceholder("system_messages", false),
	)
	msgs, err := tpl.Format(ctx, map[string]any{
		"system_messages": []*schema.Message{schema.SystemMessage(content)},
	})
	if err != nil {
		return "", fmt.Errorf("entity prompt callbacks: %w", err)
	}
	if len(msgs) == 0 || msgs[0] == nil {
		return "", fmt.Errorf("entity prompt callbacks: empty result")
	}
	return msgs[0].Content, nil
}

type EntitySpan struct {
	Type       string
	Raw        string
	Start      int
	End        int
	Confidence float64
}

type EntityOutput struct {
	Entities []EntitySpan
	Missing  []string
	Language string
}

// EntitiesByType returns all entities of a given type
func (o *EntityOutput) EntitiesByType(t string) []EntitySpan {
	out := []EntitySpan{}
	for _, e := range o.Entities {
		if e.Type == t {
			out = append(out, e)
		}
	}
	return out
}

// MissingKeys recomputes missing keys given required list
func (o *EntityOutput) MissingKeys(required []string) []string {
	miss := []string{}
	found := map[string]bool{}
	for _, e := range o.Entities {
		if e.Confidence > 0.5 && e.Start >= 0 {
			found[e.Type] = true
		}
	}
	for _, k := range required {
		if !found[k] {
			miss = append(miss, k)
		}
	}
	return miss
}

// ParseEntityOutput parses raw LLM output into EntityOutput
func ParseEntityOutput(raw string) (*EntityOutput, error) {
	out := &EntityOutput{}
	lines := strings.Split(raw, "##")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || line == "<|COMPLETE|>" {
			continue
		}

		fields := strings.Split(line, "<||>")
		if len(fields) == 0 {
			continue
		}

		switch {
		case strings.HasPrefix(fields[0], "(entity"):
			if len(fields) < 6 {
				continue
			}
			start, _ := strconv.Atoi(fields[3])
			end, _ := strconv.Atoi(fields[4])
			conf, _ := strconv.ParseFloat(fields[5], 64)
			out.Entities = append(out.Entities, EntitySpan{
				Type:       fields[1],
				Raw:        fields[2],
				Start:      start,
				End:        end,
				Confidence: conf,
			})

		case strings.HasPrefix(fields[0], "(missing"):
			if len(fields) < 2 {
				continue
			}
			out.Missing = append(out.Missing, fields[1])

		case strings.HasPrefix(fields[0], "(language"):
			if len(fields) < 4 {
				continue
			}
			out.Language = fields[1]
		}
	}

	return out, nil
}

// RequiredKeysForIntent reads required entity keys for a given intent from environment variables.
// Looks up NLU_REQUIRED_<INTENT_NAME_IN_UPPER_SNAKE> and splits comma-separated values.
func RequiredKeysForIntent(intentName string) []string {
	if intentName == "" {
		return nil
	}
	normalized := strings.TrimSpace(intentName)
	replacer := strings.NewReplacer(" ", "_", "-", "_")
	envKey := "NLU_REQUIRED_" + strings.ToUpper(replacer.Replace(normalized))
	value := os.Getenv(envKey)
	if value == "" {
		return nil
	}
	parts := strings.Split(value, ",")
	keys := make([]string, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			keys = append(keys, trimmed)
		}
	}
	return keys
}
