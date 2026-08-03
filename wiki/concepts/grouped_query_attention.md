---
id: grouped_query_attention
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [分组查询注意力, GQA]
related_nodes: [ainslie_2023_gqa, multi_head_attention, multi_head_latent_attention]
last_verified: 2026-08-03
---

# Grouped Query Attention (GQA)

## 定义
介于多头注意力（MHA）与多查询注意力（MQA）之间的折中：查询保持多头，键值按组共享，将 KV cache 缩减为 MHA 的 1/组数，质量接近 MHA、速度接近 MQA。

## 关键要点
- **共享粒度**：8 组 KV 头共享是常见配置（LLaMA-2/3、Mistral 采用）
- **Uptraining**：可从 MHA 检查点转换，无需重新训练
- **与 MLA 对比**：MLA 用潜在向量压缩 KV，GQA 用分组共享——两条 KV 缩减路线

## 来源
- [[ainslie_2023_gqa]] — GQA 原始论文
