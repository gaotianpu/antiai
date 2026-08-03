---
id: ainslie_2023_gqa
type: source
tags: ["machine-learning", "empirical-study"]
aliases: ["GQA", "Grouped Query Attention", "2305.13245"]
related_nodes: [grouped_query_attention, multi_head_attention]
authors: Joshua Ainslie et al.
authors_institution: Google
arxiv_id: 2305.13245
last_verified: 2026-08-03
---

# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

- **元数据**: arXiv 2305.13245 | 2023 | **作者**: Joshua Ainslie et al. | **机构**: Google
- **概述**: 提出分组查询注意力（GQA）：查询保持多头、键值分组共享，介于 MHA 与 MQA 之间，以接近 MHA 的质量获得接近 MQA 的推理速度。
- **新颖概念**: [[grouped_query_attention]]
- **关键要点**: 1. KV 头分组共享（如 8 组），KV cache 缩减约 1/组数 2. 可从 MHA 检查点 Uptraining 转换 3. 质量-速度权衡优于 MQA
- **方法/发现**: 在 T5 系列上验证：GQA 质量接近 MHA，推理快于 MHA；LLaMA-2/3、Mistral 等主流模型采用
- **局限/意义**: 与 MLA（潜在压缩）是 KV cache 缩减的两条路线；GQA 因实现简单成为开源 LLM 事实标准

## 引用
- **原始论文**: [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- **相关概念**: [[grouped_query_attention]], [[multi_head_attention]], [[multi_head_latent_attention]]
