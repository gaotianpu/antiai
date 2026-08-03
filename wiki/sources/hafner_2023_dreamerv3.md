---
id: hafner_2023_dreamerv3
type: source
tags: ["machine-learning", "empirical-study", "RL"]
aliases: ["DreamerV3", "Mastering Diverse Domains through World Models", "2301.04104"]
related_nodes: [world_model, model_based_rl]
authors: Danijar Hafner et al.
authors_institution: Google
arxiv_id: 2301.04104
last_verified: 2026-08-03
---

# Mastering Diverse Domains through World Models

- **元数据**: arXiv 2301.04104 | 2023 | **作者**: Danijar Hafner et al. | **机构**: Google
- **概述**: 提出 DreamerV3，基于世界模型的强化学习智能体：在 150+ 任务（Atari、Minecraft、机器人）上以固定超参数达到 SOTA，首个在 Minecraft 中从零收集钻石的智能体。
- **新颖概念**: [[world_model]], [[model_based_rl]]
- **关键要点**: 1. 学习潜在空间世界模型，在想象中训练策略 2. 符号化奖励归一化（symlog）解决奖励尺度差异 3. 固定超参数跨任务通用，无需调参
- **方法/发现**: 世界模型预测潜在状态与奖励 → actor-critic 在想象轨迹上学习；鲁棒性来自归一化技巧与模型规模
- **局限/意义**: 基于模型的 RL 里程碑，证明世界模型路线的通用性；为后续"世界模型 + 规划"研究（如 Genie）奠定基础

## 引用
- **原始论文**: [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- **相关概念**: [[world_model]], [[model_based_rl]]
