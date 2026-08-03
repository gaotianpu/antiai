---
id: bert
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [BERT, 双向编码器表示]
related_nodes: [devlin_2018_bert, masked_language_modeling, transformer_architecture]
last_verified: 2026-08-03
---

# BERT

## 定义
Bidirectional Encoder Representations from Transformers：深度双向 Transformer 编码器，通过 MLM + NSP 预训练后微调，统治 11 项 NLP 任务。

## 关键要点
- **双向性**：MLM 掩码预测使每个位置同时看到左右上下文（区别于 GPT 的单向）
- **预训练-微调范式**：通用预训练 + 下游微调，成为 NLP 标准流程
- **影响**：开启了预训练语言模型时代，衍生 RoBERTa/ALBERT/ELECTRA 等大量变体

## 来源
- [[devlin_2018_bert]] — BERT 原始论文
