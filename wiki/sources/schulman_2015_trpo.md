---
id: schulman_2015_trpo
type: source
tags: ["machine-learning", "theoretical", "RL"]
aliases: ["TRPO", "Trust Region Policy Optimization", "1502.05477"]
related_nodes: ["schulman_2017_ppo"]
arxiv_id: 1502.05477
authors: John Schulman et al.
authors_institution: UC Berkeley
last_verified: 2026-06-06
---

# Trust Region Policy Optimization

- **元数据**: arXiv | 2015 | **作者**: John Schulman et al. | **机构**: UC Berkeley | 相关: [[schulman_2017_ppo]]
- **概述**: 提出信任区域策略优化，通过 KL 散度约束保证策略更新的单调改进。
- **新颖概念**: [[trust_region_method]]
- **关键要点**: 1. 自然梯度 + 信任区域 2. 单调改进保证 3. 二阶优化
- **方法/发现**: 用 KL 散度约束替代学习率调节策略更新步长
- **局限/意义**: 奠定了策略梯度方法的理论保障，PPO 是其简化版

## 引用
- **原始论文**: [arXiv:1502.05477](https://arxiv.org/abs/1502.05477) | [阅读笔记](../../raw/RL/TRPO.md)
- **相关概念**: [[schulman_2017_ppo]]
