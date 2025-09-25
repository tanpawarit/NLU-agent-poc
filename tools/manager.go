package tools

// import (
// 	"context"

// 	"github.com/cloudwego/eino/components/tool"
// 	"github.com/cloudwego/eino/schema"
// )

// func GetQueryTools() []tool.BaseTool {
// 	return []tool.BaseTool{
// 		createSearchArticleTool(),
// 	}
// }

// // GetToolInfos extracts ToolInfo from all tools
// func GetToolInfos(ctx context.Context, tools []tool.BaseTool) ([]*schema.ToolInfo, error) {
// 	infos := make([]*schema.ToolInfo, len(tools))
// 	for i, t := range tools {
// 		info, err := t.Info(ctx)
// 		if err != nil {
// 			return nil, err
// 		}
// 		infos[i] = info
// 	}
// 	return infos, nil
// }
