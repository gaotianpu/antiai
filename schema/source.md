# Source 页创建规范

Source 页 = 论文/报告摘要卡片。见 AGENTS.md §wiki/ 页面类型速查。

---

## 页面结构

元数据统一使用 YAML frontmatter，字段必填项见 `schema/frontmatter.md`。

```markdown
---
id: [姓_年份_关键词]
type: source
tags: [方法论tag, 技术方法tag]
aliases: [中文名, 英文缩写, arXiv ID]
related_nodes: [涉及的概念/实体 page id]
arxiv_id: [arXiv 论文 ID]   # 可选，用于精确去重
authors: [第一作者 et al. / 作者列表]
authors_institution: [机构1, 机构2]   # 可选，作者所属机构
---

# [完整标题]

- **元数据**: [arXiv | 会议 | 期刊] | [年份] | **作者**: [作者] | **机构**: [机构] | 相关: [[概念/实体]]
- **概述**: [100 字内核心贡献]
- **关键要点**: 1. [要点A] 2. [要点B] 3. [要点C]
- **方法/发现**: [研究手段与主要结论]
- **局限/意义**: [局限性及实操建议]
```

## 规范

| 维度 | 要求 |
|:---|:---|
| **字数** | 正文限 500 字内，不展开论文全部细节 |
| **关键要点** | 3-6 条，每条一句话 |
| **概述** | 100 字内，让读者 10 秒决定是否值得深入 |
| **引用** | 内部链接用 `[[page_id]]`，外部文件用 `[文本](路径)` |
| **related_nodes** | 仅列已有的 wiki 页面 id，不列未创建页面 |

## 与 raw/ 的关联

Source 页通过 `arxiv_id` 字段与 `raw/` 下的原始资料建立对应关系：

```
wiki/sources/[id].md        ← 摘要卡片（本文）
       ↕ arxiv_id 字段
raw/{arxiv_id}.md           ← 转换清理后的 Markdown
raw/pdf/{arxiv_id}.pdf      ← 原始 PDF
```

示例：`wiki/sources/cheng_2026_engram.md` 的 `arxiv_id: 2601.07372` 指向 `raw/2601.07372.md` 和 `raw/pdf/2601.07372.pdf`。

## 文件命名

`wiki/sources/[姓_年份_关键词].md`，蛇形命名，全小写。

示例：`cheng_2026_engram.md`

## 去重与冲突处理

| 场景 | 处理方式 |
|:---|:---|
| **同作者同年多篇** | 关键词足够区分时保留；无法区分时加字母后缀，如 `cheng_2026_engram.md` / `cheng_2026_moe.md` |
| **arXiv 版 + 会议版** | 视为同一篇，不另建新页。先在已有页的元数据中补充会议信息，再更新内容 |
| **arXiv ID 碰撞检测** | 创建前执行：`grep -r "arxiv_id:.*XXXX" wiki/sources/`。若命中，说明已入库，应更新而非新建 |

## 防重复流程

创建 Source 页前，执行以下检查：

```bash
# 1. arXiv ID 精确碰撞
ARXIV_ID="2601.07372"
if grep -qr "arxiv_id:.*$ARXIV_ID" wiki/sources/; then
  echo "WARN: arXiv ID $ARXIV_ID already exists in wiki/sources/. Update instead of create."
  exit 1
fi

# 2. 标题近似碰撞（可选，用于非 arXiv 来源）
grep -ril "$(head -1 raw/$ARXIV_ID.md | cut -c1-60)" wiki/sources/ 2>/dev/null \
  && echo "WARN: similar title found. Possible duplicate."
```
