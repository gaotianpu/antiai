---
id: proximal_policy_optimization
type: concept
tags: [machine-learning, empirical-study, RL]
aliases: [近端策略优化, PPO]
related_nodes: [schulman_2017_ppo, trust_region_method, policy_gradient]
last_verified: 2026-08-03
---

# Proximal Policy Optimization (PPO)

## 定义
通过裁剪（clip）重要性采样比将策略更新限制在近端区域的算法：$\min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)$，兼具 TRPO 的稳定性与一阶实现的简单性。

## 关键要点
- **裁剪目标**：比值偏离 [1-ε, 1+ε] 时梯度置零，防止大步更新
- **易用性**：只需一阶优化，工业界事实标准
- **地位**：RLHF 对齐、游戏 AI（OpenAI/DeepMind）的主力算法

## 来源
- [[schulman_2017_ppo]] — PPO 原始论文
