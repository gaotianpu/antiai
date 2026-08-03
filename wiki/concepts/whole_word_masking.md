---
id: whole_word_masking
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [全词掩码, WWM]
related_nodes: [bert_wwm_2019, masked_language_modeling, bert]
last_verified: 2026-08-03
---

# Whole Word Masking (WWM)

## 定义
MLM 的掩码粒度变体：当子词被选中时，同时掩码其所属完整词的所有子词，而非仅掩码单个子词片段。

## 关键要点
- **动机**：原始 MLM 掩码子词片段，预测任务过于简单且与整词语义脱节
- **效果**：中文等无空格语言的预训练质量显著提升
- **应用**：BERT-wwm、RoBERTa-wwm 等中文预训练模型的标准配置

## 来源
- [[bert_wwm_2019]] — 提出全词掩码中文 BERT
