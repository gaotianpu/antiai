---
id: deepseek_2024_v3
type: source
tags: [empirical-study, NLP, machine-learning]
aliases: ["DeepSeek-V3", "DeepSeek-V3 Technical Report", "arXiv 2412.19437"]
related_nodes: [deepseek_ai, mixture_of_experts, multi_head_latent_attention, multi_token_prediction]
arxiv_id: 2412.19437
authors: DeepSeek-AI
authors_institution: DeepSeek
---

# DeepSeek-V3 Technical Report

- **元数据**: arXiv | 2024 | **作者**: DeepSeek-AI | **机构**: DeepSeek | 相关: [[deepseek_ai]], [[mixture_of_experts]]
- **概述**: 671B MoE（37B 激活）语言模型，以 $5.6M 低成本训练达到匹敌 GPT-4o 的性能。
- **新颖概念**: [[multi_head_latent_attention]], [[multi_token_prediction]], auxiliary-loss-free load balancing
- **关键要点**: 1. 无辅助损失负载均衡策略 2. Multi-Token Prediction 训练目标 3. FP8 混合精度训练 + 4D 并行
- **方法/发现**: 在 14.8T token 上预训练，全程无回滚。2.788M H800 GPU hours 完成训练。
- **局限/意义**: 开源 MoE 旗舰，验证了低成本训练 SOTA 模型的可行性。
