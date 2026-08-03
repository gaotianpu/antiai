---
id: experience_replay
type: concept
tags: [machine-learning, empirical-study, RL]
aliases: [经验回放, 回放缓冲区]
related_nodes: [mnih_2013_dqn, deep_q_network]
last_verified: 2026-08-03
---

# Experience Replay

## 定义
将交互经验（状态/动作/奖励/下一状态）存入缓冲区，训练时随机采样小批量的技术，打破样本时序相关性并提高数据利用率。

## 关键要点
- **去相关**：随机采样消除连续样本的强相关性，稳定训练
- **复用**：每条经验可被多次学习，样本效率提升
- **变体**：优先经验回放（PER）按 TD 误差加权采样

## 来源
- [[mnih_2013_dqn]] — DQN 的关键稳定技术
