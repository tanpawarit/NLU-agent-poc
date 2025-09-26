package intent

import (
	"encoding/json"
	"fmt"
	"strings"
)

func ParseIntentOutput(raw string) (*IntentOutput, error) {
	out := &IntentOutput{}
	lines := strings.Split(raw, "##")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || line == "<|COMPLETE|>" {
			continue
		}

		fields := strings.Split(line, "<||>")
		if len(fields) < 5 {
			continue
		}

		prefix := fields[0]
		switch prefix {
		case "(intent":
			confidence, priority := parseFloat(fields[2]), parseFloat(fields[3])
			meta := make(map[string]any)
			_ = json.Unmarshal([]byte(fields[4]), &meta)
			out.Intents = append(out.Intents, IntentResult{
				Name:       fields[1],
				Confidence: confidence,
				Priority:   priority,
				Meta:       meta,
			})
		case "(language":
			confidence := parseFloat(fields[2])
			primary := int(parseFloat(fields[3]))
			meta := make(map[string]any)
			_ = json.Unmarshal([]byte(fields[4]), &meta)
			out.Languages = append(out.Languages, LanguageResult{
				Code:        fields[1],
				Confidence:  confidence,
				PrimaryFlag: primary,
				Meta:        meta,
			})
		}
	}

	return out, nil
}

func parseFloat(s string) float64 {
	var f float64
	fmt.Sscanf(s, "%f", &f)
	return f
}
