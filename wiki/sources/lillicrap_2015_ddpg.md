---
id: lillicrap_2015_ddpg
type: source
tags: ["machine-learning", "empirical-study", "RL"]
aliases: ["DDPG", "Deep Deterministic Policy Gradient", "1509.02971"]
related_nodes: ["mnih_2013_dqn"]
arxiv_id: 1509.02971
authors: Timothy Lillicrap et al.
authors_institution: DeepMind
last_verified: 2026-06-06
---

# Continuous Control with Deep Reinforcement Learning

- **元数据**: ICLR | 2015 | **作者**: Timothy Lillicrap et al. | **机构**: DeepMind | 相关: [[mnih_2013_dqn]]
- **概述**: 将 DQN 思路扩展到连续动作空间，结合 Actor-Critic 和 DQN 的经验回放 + Target Network。
- **新颖概念**: [[actor_critic]], [[deterministic_policy_gradient]]
- **关键要点**: 1. 确定性策略梯度 2. Actor-Critic 架构 3. 连续动作空间 4. 经验回放 + Batch Normalization
- **方法/发现**: DPG + DQN 技巧，首次在连续控制任务上达到 competitive 结果
- **局限/意义**: 连续控制深度 RL 的标准方法

## 引用
- **原始论文**: [arXiv:1509.02971](https://arxiv.org/abs/1509.02971) | [阅读笔记](../../raw/RL/DDPG.md)
- **相关概念**: [[mnih_2013_dqn]]
