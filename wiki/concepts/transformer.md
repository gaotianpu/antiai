---
id: transformer
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [Transformer, 变换器]
related_nodes: [vaswani_2017_transformer, transformer_architecture, multi_head_attention, self_attention]
last_verified: 2026-08-03
---

# Transformer

## 定义
完全基于注意力机制的序列建模架构，摒弃循环与卷积，通过自注意力并行建模任意距离依赖，是现代 LLM 的基础范式。

## 关键要点
- **并行化**：注意力可并行计算，训练效率远超 RNN
- **三大组件**：自注意力、位置编码、前馈网络
- **影响**：GPT/BERT/T5 等全部主流预训练模型均基于此架构

## 来源
- [[vaswani_2017_transformer]] — "Attention Is All You Need"
- [[bahdanau_2014_attention]] — 注意力机制先驱
