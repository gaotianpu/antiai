---
id: glorot_2010_xavier
type: source
tags: ["machine-learning", "theoretical"]
aliases: ["Xavier Init", "Glorot Init", "归一化初始化"]
related_nodes: []
authors: Xavier Glorot et al.
authors_institution: University of Montreal
last_verified: 2026-06-06
---

# Understanding the Difficulty of Training Deep Feedforward Neural Networks

- **元数据**: PMLR | 2010 | **作者**: Xavier Glorot et al. | **机构**: University of Montreal
- **概述**: 分析深层网络训练困难的原因，提出 Xavier/Glorot 初始化，在前向和反向传播中维持激活方差恒定。
- **新颖概念**: [[xavier_initialization]]
- **关键要点**: 1. 激活值方差随层数衰减/爆炸 2. 归一化初始化保持方差恒定 3. 配合 tanh 激活效果最优
- **方法/发现**: 初始化权重方差 = 2/(fan_in + fan_out)
- **局限/意义**: 深度学习初始化标准方法，后续 He Init 的基准

## 引用
- **原始论文**: [PMLR](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) | [阅读笔记](../../raw/Xavier_init.md)
