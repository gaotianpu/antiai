---
id: wang_2016_acer
type: source
tags: ["machine-learning", "empirical-study", "RL"]
aliases: ["ACER", "Sample Efficient Actor-Critic with Experience Replay", "1611.01224"]
related_nodes: ["lillicrap_2015_ddpg", "mnih_2016_a2c"]
arxiv_id: 1611.01224
authors: Ziyu Wang et al.
authors_institution: DeepMind
last_verified: 2026-06-06
---

# Sample Efficient Actor-Critic with Experience Replay

- **元数据**: arXiv | 2016 | **作者**: Ziyu Wang et al. | **机构**: DeepMind | 相关: [[lillicrap_2015_ddpg]], [[mnih_2016_a2c]]
- **概述**: 将经验回放引入 Actor-Critic，结合 Retrace 和截断重要性采样，提升样本效率。
- **新颖概念**: —
- **关键要点**: 1. 经验回放 + Actor-Critic 2. Retrace 评估 3. 截断重要性采样
- **方法/发现**: 解决 off-policy 训练中高方差问题，Atari 上显著优于 A3C
- **局限/意义**: off-policy Actor-Critic 的奠基工作

## 引用
- **原始论文**: [arXiv:1611.01224](https://arxiv.org/abs/1611.01224) | [阅读笔记](../../raw/RL/ACER.md)
- **相关概念**: [[lillicrap_2015_ddpg]], [[mnih_2016_a2c]]
