---
id: asynchronous_learning
type: concept
tags: [machine-learning, empirical-study, RL]
aliases: [异步学习, 异步训练]
related_nodes: [mnih_2016_a2c, actor_critic]
last_verified: 2026-08-03
---

# Asynchronous Learning

## 定义
A3C 提出的并行训练框架：多个 Actor 线程各自与环境交互并异步更新共享参数，并行样本本身去相关，替代经验回放。

## 关键要点
- **并行探索**：多线程多样本自然去相关，无需回放缓冲区
- **Actor-Learner**：每线程既是 Actor 也是 Learner，异步梯度更新
- **意义**：早期深度 RL 的并行范式，后被同步版 A2C 简化

## 来源
- [[mnih_2016_a2c]] — A3C/A2C
