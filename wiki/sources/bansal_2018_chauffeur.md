---
id: bansal_2018_chauffeur
type: source
tags: ["autonomous-driving", "imitation-learning", "behavior-cloning"]
aliases: ["ChauffeurNet", "1812.03079"]
related_nodes: []
arxiv_id: 1812.03079
authors: Mayank Bansal et al.
authors_institution: Google (Waymo)
last_verified: 2026-06-06
---

# ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst

- **元数据**: arXiv | 2018 | **作者**: Mayank Bansal et al. | **机构**: Google (Waymo)
- **概述**: 通过模仿学习和合成扰动数据训练鲁棒的自动驾驶策略
- **关键要点**: 1. 标准行为克隆即使有 3000 万样本也不足以处理复杂驾驶场景 2. 提出向学习者暴露合成扰动数据（碰撞/偏离道路）来增强鲁棒性 3. 除模仿损失外，增加惩罚不良事件和鼓励进展的额外损失
- **方法/发现**: 中等级别输入/输出表示 + RNN + 数据扰动增强
- **局限/意义**: 证明了纯模仿在驾驶领域的局限性，合成数据扰动是提升鲁棒性的有效途径

## 引用
- **原始论文**: [arXiv:1812.03079](https://arxiv.org/abs/1812.03079) | [阅读笔记](../../raw/Autonomous_Robot/ChauffeurNet.md)
