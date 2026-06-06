---
id: mnih_2016_a2c
type: source
tags: ["machine-learning", "empirical-study", "RL"]
aliases: ["A3C", "A2C", "Asynchronous Methods for Deep RL", "1602.01783"]
related_nodes: ["mnih_2013_dqn"]
arxiv_id: 1602.01783
authors: Volodymyr Mnih et al.
authors_institution: DeepMind
last_verified: 2026-06-06
---

# Asynchronous Methods for Deep Reinforcement Learning

- **元数据**: ICML | 2016 | **作者**: Volodymyr Mnih et al. | **机构**: DeepMind | 相关: [[mnih_2013_dqn]]
- **概述**: 提出异步多线程训练框架（A3C/A2C），通过并行 Actor 替代经验回放，稳定训练。
- **关键要点**: 1. 多线程异步 Actor 2. 无需经验回放 3. A3C（异步）+ A2C（同步）变体
- **方法/发现**: CPU 多线程并行训练，消除对 GPU 经验回放的依赖
- **局限/意义**: 简化深度 RL 训练流程，A2C 成为主流 Actor-Critic 基线

## 引用
- **原始论文**: [arXiv:1602.01783](https://arxiv.org/abs/1602.01783) | [阅读笔记](../../raw/RL/A2C.md)
- **相关概念**: [[mnih_2013_dqn]]
