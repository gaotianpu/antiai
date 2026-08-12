---
name: agent-doc-gather
description: 将散落在各项目中的 agent 所需文档进行收集、归集、整理与维护。扫描 workspace 和 agents 目录，建立 README 文件映射(file_mapping.md)、检测副本变更、对比版本新旧。提示词：文档收集、文档归集、文档整理、跨项目文档、散落文档、gather docs
tags: [收集, 归集, 跨项目, 文档追踪]
aliases: [doc-gather, doc-collect, doc-harvest, 文档收集]
---

# Doc Gather — 跨项目文档归集

将散落在各项目（workspace、.agents）中的 agent 所需文档进行收集、归集、整理与维护。
产出 `file_mapping.md`，记录 README.md 中列出的文件与实际文件路径的一对多映射，并检测副本变更、对比版本新旧。

---

## 前置条件

- 当前工作目录为本项目根（`~/workspace/agent`）
- `~/workspace/agent/README.md` 存在且包含 `## rules/ — 项目约束与指南` 表格
- 扫描路径 `~/workspace`、`~/.agents` 可读

---

## 扫描范围

| 路径 | 匹配规则 |
|------|---------|
| `~/workspace/` | 完整路径包含 `/rules/` 的文件，**排除本项目自身** |
| `~/.agents/` | 全部文件（递归） |
| `~/workspace/.agents/` | 全部文件（递归） |

## README 参照

以 `~/workspace/agent/README.md` 为准，取其 `## rules/ — 项目约束与指南` 表格中的文件名（第一列 `` `xxx.md` ``）作为映射索引键。

---

## 执行

直接执行外提脚本（不依赖 CWD，脚本自动定位项目根）：

```bash
# 仅报告（默认）
bash .agents/skills/agent-doc-gather/run.sh

# 报告 + 将变更同步到本项目
bash .agents/skills/agent-doc-gather/run.sh --apply
```

`--apply` 模式：当检测到其他位置有 🔴 已修改 或 🟢 新增 的文件时，自动将其复制到本项目对应路径下（workspace 中其他项目的 `rules/` 映射到本项目 `rules/`，`.agents/skills/` 映射到本项目 `skills/`）。

该脚本自动完成：
1. 扫描 `~/workspace/`（排除本项目）、`~/.agents/`、`~/workspace/.agents/`
2. 解析 README rules 表 + skills 表
3. 构建 `file_mapping.md`（Rules 映射 + Skills 映射 + 孤儿文件）
4. 报告 A：其他项目有但本项目未收录的文档
5. 报告 B：已收录文档的版本新旧对比（mtime）
6. 基于 SHA256 快照的变更检测（新增 / 修改 / 删除）

### 输出模板

生成的 `file_mapping.md` 分三个区域：

```markdown
# File Mapping

> 最后更新: 2026-06-14 12:00:00

## Rules 文件

### unix_pythonic_guid.md

| # | 路径 |
|---|------|
| 1 | ~/workspace/research/docs/builder/rules/unix_pythonic_guid.md |
| 2 | ~/workspace/pi/templates/python_project/docs/rules/unix_pythonic_guid.md |

## Skills

### doc-fullreview/SKILL.md

| # | 路径 |
|---|------|
| 1 | ~/.agents/skills/doc-fullreview/SKILL.md |

### ⚠️ 不属于 README 的 Skill

| # | Skill | 路径 |
|---|-------|------|
| 1 | my-custom-skill | ~/.agents/skills/my-custom-skill/SKILL.md |
```

---

## 报告环节

执行时会自动输出两份报告：

### 报告 A：未收录的文档

检查其他项目的 `rules/` 目录中是否存在本项目 `rules/` 尚未收录的文件，用于发现潜在的新文档需求。

### 报告 B：版本新旧对比

对每个在 README 中且有多个副本的文档，按文件修改时间（mtime）对比各副本，标记最新版本所在位置。如果最新版本不在本项目，会列出详细对比表。

---

## 变更检测说明

首次执行：全部文件标记为 🟢 新增，快照写入 `~/.cache/fm_snapshot/`。

后续执行对比快照：

| 标识 | 含义 |
|:---:|------|
| 🟢 新增 | 快照中不存在此文件（新文件） |
| 🔴 已修改 | SHA256 与快照不一致，同时输出 unified diff |
| 🗑️ 已移除 | 快照存在但文件已不存在 |

当检测到 🔴 已修改时，会自动计算并输出 unified diff（最多 100 行），精确展示新增/删除/修改的内容。
同时会将详细 diff 报告写入 `change_logs/doc_diff_{时间戳}.md`，方便追溯历史变更。

---

## 禁止事项

- 不修改 `~/workspace/`、`~/.agents/`、`~/workspace/.agents/` 下的任何文件
- 不删除快照目录（`~/.cache/fm_snapshot/`）中的内容
- 不对扫描结果做业务判断（仅报告文件存在性和变更）

---

## 自检清单

- [ ] `README.md` 的 `## rules/ — 项目约束与指南` 表格是否存在、格式是否匹配
- [ ] 扫描结果中的文件路径是否可读
- [ ] 孤儿文件是否确实不属于 README（人工判断）
- [ ] `file_mapping.md` 的映射关系是否准确（随机抽检 2-3 条）
