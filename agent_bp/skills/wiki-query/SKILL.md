---
name: wiki-query
description: 响应用户的提问或调研需求，基于 wiki 内容生成带引用的回答，并将有价值的回答沉淀为 synthesis 页。适用于查询、对比分析、知识检索等场景。
metadata:
  variables:
    - WIKI_DIR: Wiki 页面目录（默认 $PROJECT_ROOT/wiki）
    - SCHEMA_DIR: 规范文件目录（默认 $PROJECT_ROOT/schema）
    - PROJECT_ROOT: 项目根目录（默认当前工作目录）
---

# Wiki Query — 知识检索与问答构建

## 配置变量

本 skill 使用 shell 环境变量实现路径参数化。使用前设置：

| 变量 | 默认值 | 说明 |
|:---|:---|:---|
| `PROJECT_ROOT` | `$PWD` | 项目根目录 |
| `WIKI_DIR` | `$PROJECT_ROOT/wiki` | wiki 页面目录 |
| `SCHEMA_DIR` | `$PROJECT_ROOT/schema` | 规范文件目录 |

执行任何操作前，先初始化变量：

```bash
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
WIKI_DIR="${WIKI_DIR:-$PROJECT_ROOT/wiki}"
SCHEMA_DIR="${SCHEMA_DIR:-$PROJECT_ROOT/schema}"
```

---

## 知识检索 (Query)

1. **路由 (Route)**：优先读取 `$WIKI_DIR/index.md` 锁定相关 Page。
2. **深度优先读取**：如果索引信息不足，读取具体的 `$WIKI_DIR/synthesis/` 或 `$WIKI_DIR/concepts/`。
3. **响应生成**：回答必须带引用，格式：`根据 [[资料名]]，...`。
4. **知识沉淀**：若回答涉及跨领域综合，**自动创建** `$WIKI_DIR/synthesis/[Subject].md`。

## 问答驱动构建 (Q&A-driven Build)

用户基于已有知识提出新问题（如"有关于 X 的论文吗？""这两种方法有什么区别？"）时的处理流程：

1. **调研 (Research)**：先搜索 wiki 内已有内容 + 外部检索（论文摘要、综述等）。
2. **回答 (Answer First)**：先给用户一个结构化的回答（表格、对比、分类等）。
3. **确认后构建 (Build After Confirmation)**：用户说"继续"或"需要"后，再执行：
   - 创建 `$WIKI_DIR/sources/` 页（每篇论文）。
   - 创建 `$WIKI_DIR/concepts/` 页（新概念）。
   - 创建或更新 `$WIKI_DIR/synthesis/` 页（综合分析）。
   - 更新所有索引。
4. **概念网完整性检查**：构建完成后，扫描新 page 中引用的概念 → 如果引用了不存在的概念页 → **询问用户是否需要创建**或**自动创建**。
