---
id: schulman_2017_ppo
type: source
tags: ["machine-learning", "empirical-study", "RL"]
aliases: ["PPO", "Proximal Policy Optimization", "近端策略优化", "1707.06347"]
related_nodes: []
arxiv_id: 1707.06347
authors: John Schulman et al.
authors_institution: OpenAI
last_verified: 2026-06-06
---

# Proximal Policy Optimization Algorithms

- **元数据**: arXiv | 2017 | **作者**: John Schulman et al. | **机构**: OpenAI
- **概述**: 提出 PPO，通过裁剪的替代目标实现稳定策略更新，兼顾 TRPO 的可靠性和实现简单性。
- **关键要点**: 1. 裁剪的替代目标（Clipped Surrogate Objective） 2. 实现简单，超参数少 3. 广泛用于 RLHF
- **方法/发现**: 限制策略更新幅度，避免 TRPO 的复杂二阶计算
- **局限/意义**: 成为 RL 领域事实标准，也是 RLHF 中 LLM 对齐的核心算法

## 引用
- **原始论文**: [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) | [阅读笔记](../../raw/RL/PPO.md)
