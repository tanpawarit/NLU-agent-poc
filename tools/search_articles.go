package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/chative-ai/server/internal/agent/model"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/cloudwego/eino/schema"
)

// ===================================
// Search Product Tool
// ===================================

type SearchProductInput struct {
	Query      string `json:"query"`
	Category   string `json:"category,omitempty"`
	MaxResults int    `json:"max_results,omitempty"`
}

type SearchProductOutput struct {
	Products []model.Product `json:"products"`
	Total    int             `json:"total"`
}

func createSearchArticleTool() tool.BaseTool {
	return utils.NewTool(
		&schema.ToolInfo{
			Name: "search_product",
			Desc: "Search for products in inventory. Supports Thai/English keywords including: มือถือ, โทรศัพท์, smartphone, phone, คอมพิวเตอร์, laptop, computer, แล็ปท็อป, โน้ตบุ๊ค. Always returns structured product data with ID, name, price, and availability. Use this tool whenever customer mentions any product.",
			ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
				"query": {
					Type:     "string",
					Desc:     "Product search keywords in Thai or English. Examples: มือถือ, smartphone, คอม, laptop, iPhone, Samsung, MacBook. Can include brand names, product types, or model numbers.",
					Required: true,
				},
				"category": {
					Type: "string",
					Desc: "Optional category filter. Available categories: smartphones, laptops, tablets, audio, wearables",
				},
				"max_results": {
					Type: "number",
					Desc: "Maximum number of products to return (default: 10, max: 20)",
				},
			}),
		},
		func(ctx context.Context, in *SearchProductInput) (*SearchProductOutput, error) {
			if in.Query == "" {
				return nil, fmt.Errorf("query is required")
			}

			if in.MaxResults == 0 {
				in.MaxResults = 10
			}

			// Search through mock products
			var matchedProducts []model.Product
			queryLower := strings.ToLower(in.Query)

			for _, product := range MockProducts {
				// Simple search matching name, category, or description
				if strings.Contains(strings.ToLower(product.Name), queryLower) ||
					strings.Contains(strings.ToLower(product.Category), queryLower) ||
					strings.Contains(strings.ToLower(product.Description), queryLower) {

					// Filter by category if specified
					if in.Category != "" && !strings.EqualFold(product.Category, in.Category) {
						continue
					}

					matchedProducts = append(matchedProducts, product)
				}
			}

			if len(matchedProducts) > in.MaxResults {
				matchedProducts = matchedProducts[:in.MaxResults]
			}

			result := &SearchProductOutput{
				Products: matchedProducts,
				Total:    len(matchedProducts),
			}

			return result, nil
		},
	)
}
