---
id: deep_q_network
type: concept
tags: [machine-learning, empirical-study, RL]
aliases: [深度Q网络, DQN]
related_nodes: [mnih_2013_dqn, experience_replay, reinforcement_learning]
last_verified: 2026-08-03
---

# Deep Q-Network (DQN)

## 定义
将深度神经网络与 Q-learning 结合：CNN 从像素学习状态价值，配合经验回放与 Target Network 稳定训练，在 49 个 Atari 游戏达到人类水平。

## 关键要点
- **Q 网络**：$Q(s,a;\theta)$ 逼近最优动作价值
- **两大稳定术**：经验回放打破数据相关性 + Target Network 冻结目标
- **意义**：深度 RL 的开创性工作，CNN+RL 范式的基础

## 来源
- [[mnih_2013_dqn]] — DQN 原始论文
