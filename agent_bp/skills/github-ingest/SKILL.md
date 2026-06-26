---
name: github-ingest
description: GitHub 项目分析入库，生成项目说明文档
---

# Skill: GitHub 项目分析入库

**触发条件**: 用户指令包含"入库github.com/owner/repo"。
**核心产出**: 
- `raw/github/owner.repo.md` — 项目分析文档（**必选**）
- `raw/github/owner.repo.<file>.md` — 项目核心灵魂文件原文存档（**推荐**）

---

## 1. 工作流 (Workflow)

### 阶段 1：获取项目信息 (Fetch)
1. **抓取 README**: 访问 `https://github.com/owner/repo`，获取项目 README 及基本元数据。
2. **抓取项目结构**: 通过 GitHub API 或页面解析获取仓库目录结构。
3. **元数据收集**: 作者、Stars、License、主要语言、项目描述。

### 阶段 2：深度分析 (Analyze)
根据获取的信息完成以下分析维度：
1. **项目定位**: 项目解决了什么问题？核心创新点是什么？
2. **架构设计**: 模块划分、核心抽象、数据流。
3. **技术栈**: 语言、框架、关键依赖。
4. **设计哲学**: 哪些设计决策值得关注。

### 阶段 3：灵魂文件识别 (Identify Soul Files)

有的项目有 1-2 个文件是其"灵魂"——承载了该项目的核心哲学或运行机制。这些文件值得原文存档，与分析文档互补。

**判断标准**：该文件是否揭示了项目"为什么这样做"的根本逻辑？典型候选：
- **`AGENTS.md` / `CLAUDE.md`** — Agent 运行指令（定义了项目如何被 AI 驱动）
- **`program.md`** — 研究纲领或 Agent 核心流程（如 karpathy/autoresearch）
- **设计文档 / 规范文件** — `specs/`、`design/` 下的文档
- **核心算法/架构说明** — 非代码注释，而是独立的设计文档

> 注意：README 不必单独存档，其内容已融入分析文档。只存档那些 README **承载不了**的深度内容。

### 阶段 4：生成文档 (Generate)

**产出 1：`raw/github/owner.repo.md`** — 项目分析文档（必选），格式如下：

```
# owner/repo 项目分析 & [与本项目的关联角度]

> 项目地址: https://github.com/owner/repo
> 作者: [author]
> 分析日期: [YYYY-MM-DD]

---

## 一、项目概述
[核心概念、设计理念、运行流程]

## 二、项目文件结构
[目录树 + 各模块说明]

## 三、[与本项目相关的分析维度]
[根据项目特点选择 1-3 个分析维度]

## 四、总结
[关键 takeaways]
```

**产出 2：`raw/github/owner.repo.<soul-file>.md`** — 灵魂文件原文存档（推荐）

将识别出的灵魂文件原文写入独立文件，文件名保留原始文件名语义。例如 `program.md` → `anomalyco.opencode.program.md`。文件头部注明出处：

```markdown
# <原始文件名>

**source**: <raw.githubusercontent.com URL>
**项目**: <owner/repo>
**说明**: <该文件在项目中的角色>
```

---

### 阶段 5：索引注册 (Register)
1. 在 `raw/github/README.md`（若存在）中追加索引行。
2. 在 `wiki/log/` 中记录操作。

---

## 2. 输出质量要求

| 维度 | 要求 |
|:---|:---|
| **准确性** | 项目事实（作者、Star 数、协议等）必须真实可查 |
| **结构性** | 必须包含 `## 一/二/三/四` 四级标题 |
| **独创性** | 不只是翻译 README，需提炼设计哲学和关键决策 |
| **关联性** | 明确说明该项目与 myDocs 知识库的关联 |

---

## 3. 示例

用户: `入库github.com/anomalyco/opencode`

产出:
- `raw/github/anomalyco.opencode.md` — 项目分析（必选）
- 可选灵魂文件如 `AGENTS.md` 等（按实际判断）

用户: `入库github.com/karpathy/autoresearch`

产出:
- `raw/github/karpathy.autoresearch.md` — 项目分析（必选）
- `raw/github/karpathy.autoresearch.program.md` — 灵魂文件：Agent 研究纲领
- `raw/github/karpathy.autoresearch.README.md` — 灵魂文件：项目概述文档
