---
id: actor_critic
type: concept
tags: [machine-learning, empirical-study, RL]
aliases: [演员评论家, Actor-Critic]
related_nodes: [mnih_2016_a2c, lillicrap_2015_ddpg, policy_gradient]
last_verified: 2026-08-03
---

# Actor-Critic

## 定义
策略梯度 RL 的双网络框架：Actor（策略网络）决定动作，Critic（价值网络）评估动作好坏，Critic 提供低方差基线引导 Actor 更新。

## 关键要点
- **分工**：Actor 学策略 π(a|s)，Critic 学价值 V(s)/Q(s,a)
- **低方差**：TD 误差替代完整回报，方差低于纯 REINFORCE
- **代表**：A2C/A3C、DDPG、PPO、SAC 均为 Actor-Critic 家族

## 来源
- [[mnih_2016_a2c]] — A2C/A3C
- [[lillicrap_2015_ddpg]] — DDPG 连续控制
