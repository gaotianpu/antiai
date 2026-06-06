---
name: wiki-ingest
description: 将 raw/ 中的文档（论文/报告/任意）摄入 wiki 知识库。提示词：论文入库, 概念补全, 链式页面, 知识抽取, 论文整理
metadata:
  variables:
    - RAW_DIR: 原始资料目录（默认 $PROJECT_ROOT/raw）
    - WIKI_DIR: Wiki 页面目录（默认 $PROJECT_ROOT/wiki）
    - SCHEMA_DIR: 规范文件目录（默认 $PROJECT_ROOT/schema）
    - CHANGELOG_DIR: 操作日志目录（默认 $WIKI_DIR/changelog）
    - PROJECT_ROOT: 项目根目录（默认当前工作目录）
---

# Wiki Ingest — raw/ → wiki/

将 `raw/` 中的文档（论文、报告、项目分析等）转化为 wiki 知识库中的结构化页面。遵循"扫描→原子化→索引→存证"闭环。

**触发信号**：raw/ 出现新文件、用户要求"整理论文"、"入库"、"继续"。

---

## 配置变量

| 变量 | 默认值 | 说明 |
|:---|:---|:---|
| `PROJECT_ROOT` | `$PWD` | 项目根目录 |
| `RAW_DIR` | `$PROJECT_ROOT/raw` | 原始资料目录 |
| `WIKI_DIR` | `$PROJECT_ROOT/wiki` | wiki 页面目录 |
| `SCHEMA_DIR` | `$PROJECT_ROOT/schema` | 规范文件目录 |
| `CHANGELOG_DIR` | `$WIKI_DIR/changelog` | 操作日志目录 |

```bash
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
RAW_DIR="${RAW_DIR:-$PROJECT_ROOT/raw}"
WIKI_DIR="${WIKI_DIR:-$PROJECT_ROOT/wiki}"
SCHEMA_DIR="${SCHEMA_DIR:-$PROJECT_ROOT/schema}"
CHANGELOG_DIR="${CHANGELOG_DIR:-$WIKI_DIR/changelog}"
```

---

## 阶段 1：扫描 (Scan)

1. 读取 `raw/` 新文件，识别文档类型（论文/报告/项目分析）
2. 提取元数据：标题、作者、日期、领域
3. 识别核心贡献、方法论、关键指标
4. 提取**实体**（人物/组织/指标名）和**概念**（理论/方法/模式）
5. 提取可量化的因子/公式（如适用）

---

## 引用规范

链接、转义、related_nodes 规则见 `AGENTS.md §2`。创建页面前参阅：

- `schema/source.md` — Source 页模板 + 去重流程
- `schema/concept.md` — Concept 页模板
- `schema/concept_dedup.md` — **创建 Concept 前必查**，避免重复定义
- `schema/entity.md` — Entity 页模板
- `schema/tags.md` — 受控 tag 词表

---

## 阶段 2：原子化写入 (Atomic Write)

> 以下模板为默认值。若 `$SCHEMA_DIR/source.md` 存在，优先使用项目自定义模板。

### 2.1 创建 Source 页

`$WIKI_DIR/sources/[ID].md`，正文限 500 字内。结构和字段说明见 `schema/source.md`。

**必填字段**：`新颖概念` 必须填写。Agent 据此识别哪些概念需要新建概念页。

**去重**：创建前先以 `arxiv_id` 精确碰撞检测是否已入库（见 `schema/source.md` §防重复流程）。

### 2.2 按需创建 Concept / Entity 页面

**从"新颖概念"字段触发**：source 页填写 `新颖概念` 后，对其中的每个 `[[concept_id]]` 检查概念页是否存在。不存在则创建，已存在则在原页追加新视角。

**判定准则**：实体在文中出现 **≥3 次** 且具有独立定义价值时方可建立单页。

**去重优先**：严禁创建内容重叠的页面，优先"更新"而非"新建"。

| 页面类型 | 去重方法 | 规范位置 |
|:---|:---|:---|
| **Concept** | 4 步流程：索引碰撞 → lookup_key 重叠检测（>50% 停）→ 语义边界判断 → 处置决策 | `schema/concept_dedup.md` |
| **Entity** | 别名碰撞检测：全名/缩写/常用变体搜索已有 `aliases` 字段 | `schema/entity.md` §4.1 |

若发现已有页面，在原页面末尾追加新视角及 `[[Source_ID]]` 引用，不新建。

**同步 Entity 贡献**：若 Source 页涉及已有实体（人物/组织），在该实体页的「关键贡献」节追加一行 `- **[[Source_ID]]**（年份）：[一句话贡献]`。实体页是"名片页"，不展开论文细节。

```markdown
# [名称]
- **定义**: [准确描述]
- **属性/特征**: [列表展示]
- **量化应用**: [基于数据/论文的具体场景]
- **公式/计算**: $Equation$ (如适用)
- **来源**: [[Source_ID]]
```

### 2.3 同步 related_nodes

新 page 创建后，检查所有被引用页面的 `related_nodes` 是否包含反向引用，确保双向完整。

### 2.4 路线图比对 (Triage)

如项目有 `$SCHEMA_DIR/paper_triage.md`，对论文运行四维评估矩阵（待补充：四维维度定义），决定是否更新 roadmap 中的交付项。

### 2.5 实证基准校验 (Reality Check)

如项目有 `$SCHEMA_DIR/reality_check.md`，检查论文的可操作发现是否与 Level A/B 数学约束或强实证规律冲突（待补充：Level A/B 约束定义）。冲突 → Source 页标记 `reality_check` 字段。

---

## 阶段 3：索引注册 (Register)

**仅追加，不重建**：
- `$WIKI_DIR/index.md`：对应分区追加 1 行链接
- `$WIKI_DIR/sources/index.md`、`concepts/index.md`、`entities/index.md`：同上
- 更新各索引的统计数据

---

## 阶段 4：操作存证 (Log)

在 `$CHANGELOG_DIR/log.$(date +%Y-%-m-%-d).md` 追加，每条包含 YAML frontmatter（`type: changelog`）便于 Graph 追踪：

```markdown
---
type: log
---
## [HH:MM] Ingest: [标题/ID]
- 新建 Source: ...
- 新建 Concept: ...
- 新建 Entity: ...
- 更新 Index: ...
- 双向链接: ...
```

---

## 阶段 5：Git 提交

```bash
git add . && git commit -m "Ingest: [简述]"
```

---

## 迭代推进模式

用户说"继续"时：

1. **锚定上一轮产出**：回顾最近 commit 创建/修改了哪些页面
2. **识别下一步**：
   - 用户明确指出的（如 "XX 论文也创建 source 页"）
   - 概念网缺口（新建 synthesis 页引用了不存在概念 → 补概念页）
   - 逻辑延伸（如参数高原 → 其他反过拟合参数寻优方法）
3. **按优先级执行**：Source 页 → Concept 页 → 更新既有页面 → 索引 → 日志 → commit

---

## 性能优化

处理多文档时：先扫描、后批量写入、最后统一更新索引。
