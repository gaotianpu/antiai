---
id: shao_2024_deepseekmath
type: source
tags: ["NLP", "machine-learning", "empirical-study", "RL"]
aliases: ["DeepSeekMath", "GRPO", "2402.03300"]
related_nodes: [grpo, chain_of_thought]
authors: Zhihong Shao et al.
authors_institution: DeepSeek
arxiv_id: 2402.03300
last_verified: 2026-08-03
---

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

- **元数据**: arXiv 2402.03300 | 2024 | **作者**: Zhihong Shao et al. | **机构**: DeepSeek
- **概述**: 提出 DeepSeekMath 7B，在 120B 数学语料上训练，以 GRPO 强化学习在竞赛数学（MATH 51.7%）超越 GPT-4 系基线；首次提出 GRPO 算法。
- **新颖概念**: [[grpo]]
- **关键要点**: 1. 提出 GRPO：组内相对奖励替代 critic 网络，显著降低 RL 训练资源 2. 数学语料构建：网页去重 + 代码-数学关联过滤 3. 组采样策略稳定训练
- **方法/发现**: 对同一 prompt 采样 G 个输出，以组内奖励均值归一化估计优势，PPO 式裁剪更新策略；无 critic 使显存与算力需求减半
- **局限/意义**: GRPO 成为 LLM 数学/代码推理 RL 的主流算法，直接支撑 DeepSeek-R1 的纯 RL 训练

## 引用
- **原始论文**: [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- **相关概念**: [[grpo]], [[chain_of_thought]]
