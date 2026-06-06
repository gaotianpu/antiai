# 经验教训 2026-06-06

本会话涉及 schema 规整、pdf-ingest、wiki 论文入库、实体页管理等全流程操作，记录了以下经验教训。

---

## 1. pdf-ingest 流程

### 1.1 markitdown 空格丢失陷阱
某些 LaTeX 生成的 PDF 文本层使用字距代替空格字符，markitdown/pdfplumber 提取后单词粘连（`WhileMixture-of-Experts(MoE)scalescapacity`）。正则 `([a-z])([A-Z])` 做后处理会误拆合法缩写（`MoE` → `Mo E`）。

**应对**: pymupdf (fitz) 提取效果好得多，已在 SKILL.md 中添加为备选方案。

### 1.2 arXiv 提交戳记干扰正文
arXiv 在首页嵌入 `arXiv:XXXX.XXXXXv1 [cs.CL] DD Mon YYYY`，夹在断词之间（`knowl-` + 戳记 + `edge`）。需检测并删除。

### 1.3 pymupdf 标题需后处理
pymupdf 输出纯文本无 `#` 标记，需用正则 `^\d+\.\s+Title` → `## Title`。但注意引用条目（`2378. Association...`）也会匹配此模式，需按编号阈值（>100）过滤。

### 1.4 工具选择原则
markitdown 输出有问题时 → 换 pymupdf，而非写补丁脚本修复。临时脚本 `01_fix_markitdown_spacing.py` 最终废弃，改为通用 `tools/pdf2md_fix.py`。

---

## 2. wiki 论文入库

### 2.1 批量入库效率
55 篇 NLP + 18 篇 Multimodal + 15 篇 ViT + 8 篇 CNN，总计 96 篇论文分 4 批入库。先扫描元数据 → 批量生成 Source 页 → 最后统一更新索引。索引重写时注意去重，避免多次 append 导致的重复条目。

### 2.2 元数据标准化
`authors` 和 `authors_institution` 字段需要统一定义。不同批次创建的 Source 页格式不一致（有的作者括号内带机构，有的只有机构名）。用脚本统一后还需处理名称标准化（`Meta` / `Facebook` / `Meta AI` → `Meta`；`百度` → `Baidu` 等）。

### 2.3 实体页自动关联
`authors_institution` 出现的每个机构都需要对应 entities/ 页面。创建实体页后需反向扫描所有 Source 页补全 `related_nodes`，确保双向一致。

---

## 3. 目录与导航

### 3.1 中文文件名违规
`最新前沿.md` 使用了中文文件名，违反 AGENTS.md 硬约束→立即改名为 `frontier.md`。新手入门文档的命名也需要注意。

### 3.2 NV 思想指导文档结构
`getting_started.md` 经历了 3 次重写：纯描述→教程式→MVP 思想（代码优先）。MVP 思路更符合学习者认知曲线——从最小可运行代码开始，每次加一个新概念。

---

## 4. 规范文档维护

### 4.1 schema source.md 缺失
创建 Source 页时发现 `schema/` 下缺少 `source.md`（有 concept.md / entity.md 但没有 source.md），模板原先内嵌在 wiki-ingest SKILL.md 中。已抽离为独立文件，并在 AGENTS.md 的 schema/ 索引中补充。

### 4.2 related_nodes 仅列已有页面
多个 Source 页的 `related_nodes` 引用了尚未创建的页面（如 `mixture_of_experts`、`bert`、`gpt` 等）。规范规定只引用已存在的 page id，未创建的用内联链接 `[[page_name]]` 在正文中引用。

### 4.3 去重规则碎片化
Source / Concept / Entity 三类页面的去重规则分散在不同文档中（`source.md` / `concept_dedup.md` / `entity.md`），已在 wiki-ingest SKILL.md 中用汇总表格统一引用。

---

## 5. 引用与关联

### 5.1 实体页贡献排序
`关键贡献` 应按年份升序排列，而非添加顺序。当前 `kaiming_he.md` 和 `microsoft.md` 顺序混乱，已规整并在 `schema/entity.md` 中追加约定。

### 5.2 Source 到 raw 的链接
Source 页的「引用」区中，内部链接用 `[[page_id]]`，raw 文件用文件路径 `[文本](路径)`。`arxiv_id` 字段是 Source 与 raw/ 之间的桥接字段。

---

## 6. 待办/遗留

- `raw/cnn/` 仍有大量 CNN 论文未摄入（AlexNet / VGG / YOLO 等 60+ 篇）
- `raw/Generative/`、`raw/Autonomous_Robot/` 等子目录尚未处理
- `mixture_of_experts` 概念页已建，但 MoE 相关论文（Switch Transformer、GShard 等）未入库
- `frontier.md` 当前仅 1 篇（Engram），随新论文入库需更新
