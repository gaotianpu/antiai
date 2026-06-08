---
id: deepseek_2024_v2
type: source
tags: [empirical-study, NLP, machine-learning]
aliases: ["DeepSeek-V2", "A Strong, Economical, and Efficient Mixture-of-Experts Language Model", "arXiv 2405.04434"]
related_nodes: [deepseek_ai, mixture_of_experts, multi_head_latent_attention]
arxiv_id: 2405.04434
authors: DeepSeek-AI
authors_institution: DeepSeek
---

# DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

- **元数据**: arXiv | 2024 | **作者**: DeepSeek-AI | **机构**: DeepSeek | 相关: [[deepseek_ai]], [[mixture_of_experts]]
- **概述**: 236B MoE（21B 激活）语言模型，KV cache 压缩 93.3%，训练成本降低 42.5%。
- **新颖概念**: [[multi_head_latent_attention]], DeepSeekMoE
- **关键要点**: 1. Multi-head Latent Attention 压缩 KV cache 93.3% 2. DeepSeekMoE 高效稀疏计算 3. 128K 上下文
- **方法/发现**: 8.1T token 预训练，SFT+RL 对齐，性能匹敌 DeepSeek 67B 的同时吞吐量提升 5.76×。
- **局限/意义**: MLA 奠定了后续 DeepSeek 模型的高效推理基础。
