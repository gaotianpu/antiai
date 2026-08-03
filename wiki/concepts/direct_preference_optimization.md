---
id: direct_preference_optimization
type: concept
tags: [machine-learning, theoretical, RL]
aliases: [直接偏好优化, DPO]
related_nodes: [rafailov_2023_dpo, reinforcement_learning_from_human_feedback, reward_modeling]
last_verified: 2026-08-03
---

# Direct Preference Optimization (DPO)

## 定义
将 RLHF 的"奖励建模 + RL 优化"合并为一步的偏好优化方法：数学上证明奖励模型可重参数化为策略本身，直接用偏好数据上的分类损失更新策略，无需显式奖励模型与 RL 采样。

## 关键要点
- **隐式奖励**：策略即奖励模型，Bradley-Terry 损失直接优化
- **成本优势**：无 PPO 的采样/ critic，训练稳定且算力需求低
- **变体家族**：KTO（无配对偏好）、IPO（过优化鲁棒）、SimPO（无参考模型）等

## 来源
- [[rafailov_2023_dpo]] — DPO 原始论文
