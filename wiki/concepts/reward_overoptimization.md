---
id: reward_overoptimization
type: concept
tags: [machine-learning, empirical-study, RL]
aliases: [奖励过优化, 奖励过拟合]
related_nodes: [gao_2022_rmoveroptimization, reward_modeling, goodhart_law]
last_verified: 2026-08-03
---

# Reward Overoptimization

## 定义
RLHF 中过度优化奖励模型分数导致真实质量先升后降的现象：RM 是真实偏好的有损近似，优化越久偏离越大（Goodhart's Law 的实证）。

## 关键要点
- **U 形曲线**：真实质量随优化强度先升后降
- **Scaling Law**：可通过规模预测最优优化强度（KL 预算）
- **应对**：限制 KL、RM 校准、过程监督

## 来源
- [[gao_2022_rmoveroptimization]] — 奖励过优化实证
