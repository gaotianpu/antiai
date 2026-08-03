---
id: span_boundary_objective
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [跨度边界目标, SBO]
related_nodes: [spanbert_2019_spanbert, span_masking]
last_verified: 2026-08-03
---

# Span Boundary Objective (SBO)

## 定义
SpanBERT 提出的辅助预训练目标：用 span 两端之外的边界 token 表示预测 span 内部每个 token，强化边界表示的信息承载。

## 关键要点
- **预测方式**：$h_{start}, h_{end}$ 与位置嵌入拼接后预测 span 内 token
- **与 MLM 互补**：SBO 关注边界，MLM 关注内容，联合训练
- **效果**：显著提升需要边界定位的任务（抽取式 QA、指代消解）

## 来源
- [[spanbert_2019_spanbert]] — SpanBERT 双目标预训练
