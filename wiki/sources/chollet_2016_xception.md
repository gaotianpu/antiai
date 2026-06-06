---
id: chollet_2016_xception
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["Xception", "1610.02357"]
related_nodes: ["szegedy_2015_inception_v3", "szegedy_2016_inception_v4"]
arxiv_id: 1610.02357
authors: Francois Chollet
authors_institution: Google
last_verified: 2026-06-06
---

# Xception: Deep Learning with Depthwise Separable Convolutions

- **元数据**: CVPR 2017 | arXiv | 2016 | **作者**: François Chollet | **机构**: Google
- **概述**: 将 Inception 模块重新解释为常规卷积与深度可分离卷积之间的中间步骤，并提出 Xception 架构——用深度可分离卷积完全替代 Inception 模块。
- **新颖概念**: [[depthwise_separable_convolution]]
- **关键要点**: 1. 论证深度可分离卷积等价于 Inception 模块中 tower 数量最大化的极限情况 2. Xception 与 Inception V3 参数数量相同但因参数使用效率更高而性能更优 3. 在 3.5 亿图像/17000 类的大规模分类任务上显著超越 Inception V3
- **方法/发现**: 深度可分离卷积将空间卷积与通道相关计算完全解耦，先对每个通道独立进行 3×3 卷积，再通过 1×1 卷积跨通道融合。
- **局限/意义**: Xception 预见了深度可分离卷积在 MobileNet 等轻量架构中的核心地位，但深度可分离卷积在 GPU 上的计算密集度较低，量产优化需定制实现。

## 引用
- **原始论文**: [arXiv:1610.02357](https://arxiv.org/abs/1610.02357) | [阅读笔记](../../raw/cnn/xception.md)
