---
id: masked_language_modeling
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [掩码语言建模, MLM]
related_nodes: [devlin_2018_bert, bert, self_supervised_learning]
last_verified: 2026-08-03
---

# Masked Language Modeling (MLM)

## 定义
自监督预训练目标：随机掩码输入 token 的一部分，训练模型根据剩余上下文预测被掩码的词，使编码器学习双向表示。

## 关键要点
- **双向上下文**：与自回归（从左到右）目标的关键区别
- **预训练-微调不匹配**：预训练有 [MASK]，微调无 [MASK]，催生 ELECTRA 等替代目标
- **扩展**：Whole Word Masking、Span Masking 等粒度变体

## 来源
- [[devlin_2018_bert]] — 提出 MLM 预训练目标
- [[bert_wwm_2019]] — 全词掩码变体
- [[spanbert_2019_spanbert]] — Span 级掩码变体
