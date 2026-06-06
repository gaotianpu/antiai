---
id: christiano_2017_rlhf
type: source
tags: ["machine-learning", "empirical-study", "RL"]
aliases: ["Deep RL from Human Preferences", "从人类偏好进行深度强化学习", "1706.03741"]
related_nodes: ["schulman_2017_ppo"]
arxiv_id: 1706.03741
authors: Paul Christiano et al.
authors_institution: UC Berkeley
last_verified: 2026-06-06
---

# Deep Reinforcement Learning from Human Preferences

- **元数据**: arXiv | 2017 | **作者**: Paul Christiano et al. | **机构**: UC Berkeley | 相关: [[schulman_2017_ppo]]
- **概述**: 提出从人类偏好中训练奖励模型，再用 RL 优化策略，实现复杂任务的对齐学习。
- **关键要点**: 1. 人类偏好 → 奖励模型 2. PPO 优化策略 3. 减少人工标注需求
- **方法/发现**: 用人类比较数据训练奖励模型替代手工设计奖励函数
- **局限/意义**: RLHF 的早期奠基工作，直接影响 InstructGPT/ChatGPT 的对齐方案

## 引用
- **原始论文**: [arXiv:1706.03741](https://arxiv.org/abs/1706.03741) | [阅读笔记](../../raw/RL/hp_RL.md)
- **相关概念**: [[schulman_2017_ppo]]
