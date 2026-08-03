---
id: multi_head_attention
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [多头注意力, MHA]
related_nodes: [vaswani_2017_transformer, self_attention, attention_mechanism, multi_head_latent_attention]
last_verified: 2026-08-03
---

# Multi-Head Attention

## 定义
将查询/键/值投影到多个子空间并行计算注意力，再拼接融合，使模型同时关注不同位置的不同表示子空间。

## 关键要点
- **并行子空间**：每个头学习不同的注意力模式（语法/指代/局部等）
- **实现**：$MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O$
- **变体**：MQA（共享 KV）、GQA（分组 KV）、MLA（潜在压缩）等效率优化

## 来源
- [[vaswani_2017_transformer]] — 提出多头注意力
