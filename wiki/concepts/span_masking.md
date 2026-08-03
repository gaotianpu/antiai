---
id: span_masking
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [跨度掩码, Span 掩码]
related_nodes: [spanbert_2019_spanbert, masked_language_modeling]
last_verified: 2026-08-03
---

# Span Masking

## 定义
MLM 的粒度变体：随机掩码连续 token 片段（span）而非独立词，迫使模型预测整个片段，学习更强的上下文建模能力。

## 关键要点
- **几何分布长度**：SpanBERT 按几何分布采样 span 长度，偏重短 span
- **效果**：对抽取式问答、指代消解等任务提升显著
- **对比**：比单 token 掩码更接近真实文本的连续缺失模式

## 来源
- [[spanbert_2019_spanbert]] — SpanBERT 预训练目标
