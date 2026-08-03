---
id: model_based_rl
type: concept
tags: [machine-learning, theoretical, RL]
aliases: [基于模型的强化学习, MBRL]
related_nodes: [hafner_2023_dreamerv3, world_model, reinforcement_learning]
last_verified: 2026-08-03
---

# Model-Based RL

## 定义
学习环境动态模型（world model）并利用其进行规划或训练策略的强化学习分支，与依赖真实交互的 model-free RL 相对，样本效率更高。

## 关键要点
- **样本效率**：在想象中学习，减少真实环境交互
- **模型误差**：世界模型不精确时策略会利用误差（model bias），是主要挑战
- **DreamerV3 路线**：潜在世界模型 + actor-critic 想象训练，固定超参数跨 150+ 任务通用

## 来源
- [[hafner_2023_dreamerv3]] — DreamerV3
