---
id: xie_2016_resnext
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["ResNeXt", "Aggregated Residual Transformations for Deep Neural Networks", "1611.05431"]
related_nodes: []
arxiv_id: 1611.05431
authors: Saining Xie et al.
authors_institution: UC San Diego, Facebook
last_verified: 2026-06-06
---

# Aggregated Residual Transformations for Deep Neural Networks

- **元数据**: arXiv:1611.05431 | CVPR 2017 | **作者**: Saining Xie et al. | **机构**: UC San Diego / Facebook
- **概述**: 提出"基数"（cardinality）维度，通过分组卷积提升模型性能
- **新颖概念**: [[group_convolution]]
- **关键要点**: 1. 提出 cardinality（转换集大小）作为除深度、宽度之外的新维度 2. 采用分组卷积实现同拓扑多分支结构 3. 增加 cardinality 比加深/加宽更有效
- **方法/发现**: 在相同计算量下，增加 cardinality 比增加深度或宽度带来更大精度提升
- **局限/意义**: 分组卷积的组数需调参；但 cardinality 思路深刻影响后续 EfficientNet 等设计

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1611.05431) | [阅读笔记](../../raw/cnn/resnext.md)
