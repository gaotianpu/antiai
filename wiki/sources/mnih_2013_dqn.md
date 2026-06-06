---
id: mnih_2013_dqn
type: source
tags: ["machine-learning", "empirical-study", "RL"]
aliases: ["DQN", "Deep Q-Network", "Human-level control through deep RL"]
related_nodes: []
authors: Volodymyr Mnih et al.
authors_institution: DeepMind
last_verified: 2026-06-06
---

# Human-Level Control Through Deep Reinforcement Learning

- **元数据**: Nature | 2015 | **作者**: Volodymyr Mnih et al. | **机构**: DeepMind
- **概述**: 提出 DQN，将深度神经网络与 Q-learning 结合，在 49 个 Atari 游戏中达到人类水平。
- **新颖概念**: [[deep_q_network]], [[experience_replay]]
- **关键要点**: 1. 经验回放打破数据相关性 2. Target Network 稳定训练 3. 端到端从像素学习
- **方法/发现**: CNN 处理游戏画面 + Q-learning + 经验回放，Atari 多数游戏超越人类
- **局限/意义**: 深度 RL 的开创性工作，CNN+RL 范式的基础

## 引用
- **原始论文**: [Nature](https://www.deepmind.com/publications/human-level-control-through-deep-reinforcement-learning) | [阅读笔记](../../raw/RL/DQN.md)
