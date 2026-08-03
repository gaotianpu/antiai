---
id: deterministic_policy_gradient
type: concept
tags: [machine-learning, empirical-study, RL]
aliases: [确定性策略梯度, DPG]
related_nodes: [lillicrap_2015_ddpg, actor_critic]
last_verified: 2026-08-03
---

# Deterministic Policy Gradient (DPG)

## 定义
针对连续动作空间的策略梯度：策略输出确定性动作 $a = μ(s)$（而非动作分布），梯度通过链式法则直接对动作求导。

## 关键要点
- **连续控制**：避免高维动作空间的采样积分
- **DDPG 组合**：DPG + DQN 技巧（经验回放、Target Network、软更新）
- **局限**：确定性策略天然缺少探索，需噪声注入

## 来源
- [[lillicrap_2015_ddpg]] — DDPG
