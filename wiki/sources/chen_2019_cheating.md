---
id: chen_2019_cheating
type: source
tags: ["autonomous-driving", "imitation-learning", "knowledge-distillation"]
aliases: ["Learning by Cheating", "1912.12294"]
related_nodes: []
arxiv_id: 1912.12294
authors: Dian Chen et al.
authors_institution: UT Austin
last_verified: 2026-06-06
---

# Learning by Cheating

- **元数据**: CoRL 2019 | 2019 | **作者**: Dian Chen et al. | **机构**: UT Austin
- **概述**: 将驾驶任务分解为"特权智能体"（可访问真值信息）和"纯视觉智能体"两阶段训练
- **新颖概念**: —
- **关键要点**: 1. 特权智能体通过观察环境真值布局"作弊"学习 2. 第二阶段的纯视觉智能体模仿特权智能体 3. 在 CARLA 基准上首次实现 100% 成功率
- **方法/发现**: 两阶段训练：privileged agent → sensorimotor agent 蒸馏
- **局限/意义**: 打破了端到端学习的性能上限，为知识蒸馏在驾驶中的应用奠定基础

## 引用
- **原始论文**: [arXiv:1912.12294](https://arxiv.org/abs/1912.12294) | [阅读笔记](../../raw/Autonomous_Robot/cheating.md)
