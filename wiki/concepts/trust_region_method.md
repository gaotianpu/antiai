---
id: trust_region_method
type: concept
tags: [machine-learning, empirical-study, RL]
aliases: [信任区域方法, TRPO]
related_nodes: [schulman_2015_trpo, policy_gradient, proximal_policy_optimization]
last_verified: 2026-08-03
---

# Trust Region Method

## 定义
TRPO 提出的策略更新约束：用 KL 散度限制新旧策略距离（信任区域），保证每次更新单调改进，避免大步长导致的策略崩溃。

## 关键要点
- **约束优化**：max 期望回报，s.t. KL(π‖π_old) ≤ δ
- **单调改进**：理论保证（surrogate + KL 约束）
- **代价**：二阶优化（共轭梯度）实现复杂，催生 PPO 简化

## 来源
- [[schulman_2015_trpo]] — TRPO
