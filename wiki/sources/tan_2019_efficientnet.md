---
id: tan_2019_efficientnet
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["EfficientNet", "1905.11946"]
related_nodes: []
arxiv_id: 1905.11946
authors: Mingxing Tan et al.
authors_institution: Google
last_verified: 2026-06-06
---

# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

- **元数据**: arXiv:1905.11946 | ICML 2019 | **作者**: Mingxing Tan et al. | **机构**: Google
- **概述**: 提出复合缩放方法，统一平衡深度、宽度和分辨率
- **新颖概念**: [[compound_scaling]]
- **关键要点**: 1. 复合缩放系数（compound coefficient）同时缩放深度/宽度/分辨率 2. 通过 NAS 搜索基线网络 EfficientNet-B0 3. EfficientNet-B7 在 ImageNet 达到 SOTA（top-1 84.4%）
- **方法/发现**: 深度、宽度、分辨率三者存在相互依赖关系，联合缩放比单独缩放更有效
- **局限/意义**: 缩放策略依赖基线搜索网络；但提出了模型缩放的重要方法论，影响广泛

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1905.11946) | [阅读笔记](../../raw/cnn/EfficientNet.md)
