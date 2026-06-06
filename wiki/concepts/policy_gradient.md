---
id: policy_gradient
type: concept
tags: [machine-learning, theoretical, empirical-study]
aliases: [策略梯度, PG]
related_nodes: [schulman_2015_trpo, schulman_2017_ppo, lillicrap_2015_ddpg, mnih_2016_a2c, reinforcement_learning]
last_verified: 2026-06-06
---

# Policy Gradient

## 定义
Policy Gradient（策略梯度）是一类直接对策略（policy）参数化并沿累积奖励的梯度方向更新的强化学习方法，无需通过价值函数间接推导策略。

## 核心公式
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot R]$$

其中 $R$ 为累积奖励，$\log \pi_\theta(a|s)$ 为策略的对数概率。

## 关键进展
- **REINFORCE** (Williams 1992) — 蒙特卡洛策略梯度，高方差
- **Actor-Critic**（[[mnih_2016_a2c]]）— 引入基线降低方差
- **TRPO**（[[schulman_2015_trpo]]）— 约束步长确保单调改进
- **PPO**（[[schulman_2017_ppo]]）— 裁剪替代目标，简单高效
- **DDPG**（[[lillicrap_2015_ddpg]]）— 确定性策略梯度用于连续控制

## 来源
- [[schulman_2015_trpo]] — Trust Region Policy Optimization
- [[schulman_2017_ppo]] — Proximal Policy Optimization
- [[mnih_2016_a2c]] — 同步 Advantage Actor-Critic
