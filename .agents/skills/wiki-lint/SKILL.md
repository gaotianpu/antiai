---
name: wiki-lint
description: 定期或按需对 wiki 执行质量检查：孤岛扫描、死链清理、冲突检测、TODOS 提取、表格转义检查、related_nodes 对称性验证。
metadata:
  variables:
    - WIKI_DIR: Wiki 页面目录（默认 $PROJECT_ROOT/wiki）
    - SCHEMA_DIR: 规范文件目录（默认 $PROJECT_ROOT/schema）
    - PROJECT_ROOT: 项目根目录（默认当前工作目录）
---

# Wiki Lint — 质量校验

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

Agent 需定期（或在 Query 失败时）自发执行 Lint 动作：

## 七项检查

1. **孤岛扫描**：寻找没有任何入链的 `.md` 文件（以 `$WIKI_DIR/` 为根）。
2. **死链清理**：检查 `[[ ]]` 链接的文件是否存在于 `$WIKI_DIR/` 下。
3. **冲突检测**：检查 `$WIKI_DIR/concepts/` 下是否存在对同一术语的不同定义。如项目有 `$SCHEMA_DIR/concept_dedup.md`，参照其规则处理。
4. **TODOS 提取**：扫描所有页面中的 `TODO` 或 `[?]` 标记，汇总至主页面。
5. **表格转义检查**：扫描所有表格中的 `[[...|...]]` 是否已正确转义为 `[[...\|...]]`。
6. **related_nodes 对称性**：抽查新创建的 page，验证双向引用是否完整。
7. **索引完整性检查**：运行 `$PROJECT_ROOT/scripts/check_index_completeness.py`，验证 `wiki/{sources,concepts,entities}/index.md` 是否收录了所有文件。定期执行或大批量摄入后执行。
