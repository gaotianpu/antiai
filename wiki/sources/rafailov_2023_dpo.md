---
id: rafailov_2023_dpo
type: source
tags: ["machine-learning", "empirical-study", "RL"]
aliases: ["DPO", "Direct Preference Optimization", "2305.18290"]
related_nodes: [direct_preference_optimization, reinforcement_learning_from_human_feedback]
authors: Rafael Rafailov et al.
authors_institution: Stanford
arxiv_id: 2305.18290
last_verified: 2026-08-03
---

# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

- **元数据**: arXiv 2305.18290 | 2023 | **作者**: Rafael Rafailov et al. | **机构**: Stanford
- **概述**: 提出 DPO，将 RLHF 的"奖励建模 + RL 优化"两阶段合并为一步：直接从偏好数据优化策略，无需显式奖励模型与采样。
- **新颖概念**: [[direct_preference_optimization]]
- **关键要点**: 1. 数学上证明奖励模型可被重参数化为策略本身 2. 目标函数等价于带隐式奖励的 KL 约束分类损失 3. 训练稳定、算力消耗远低于 PPO
- **方法/发现**: 偏好对上的 Bradley-Terry 损失直接更新策略，实验表明在摘要、对话等任务上与 RLHF 相当甚至更优
- **局限/意义**: 依赖偏好数据质量、分布外风险未完全解决；开启"无 RL 的对齐"研究潮（KTO/IPO/SimPO 等变体），成为开源社区对齐的事实标准

## 引用
- **原始论文**: [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- **相关概念**: [[direct_preference_optimization]], [[reinforcement_learning_from_human_feedback]]
