---
id: chen_2017_deeplabv3
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["DeepLab v3", "DeepLabv3", "1706.05587"]
related_nodes: ["resnet_2015", "sppnet_2014", "long_2014_fcn"]
arxiv_id: 1706.05587
authors: Liang-Chieh Chen et al.
authors_institution: Google
last_verified: 2026-06-06
---

# Rethinking Atrous Convolution for Semantic Image Segmentation

- **元数据**: arXiv | 2017 | **作者**: Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam | **机构**: Google
- **概述**: 系统地重新审视空洞卷积（Atrous Convolution）在语义分割中的应用，改进空洞空间金字塔池化（ASPP）模块，集成图像级特征编码全局上下文，在 PASCAL VOC 2012 上达到 SOTA 性能。
- **新颖概念**: [[atrous_convolution]], [[atrous_spatial_pyramid_pooling]]
- **关键要点**: 1. 级联和并行的空洞卷积均可捕获多尺度上下文，设计更优的级联模块和 ASPP 模块 2. 在 ASPP 中引入图像级特征（全局平均池化）作为全局上下文增强 3. 无需 DenseCRF 后处理仍取得与之前 DeepLab 版本相当的优异结果
- **方法/发现**: 使用 ResNet 作为骨干，探索不同空洞率的组合对分割质量的影响，找到最优多尺度配置。
- **局限/意义**: DeepLabv3 是空洞卷积用于语义分割的集大成之作。后续 DeepLabv3+ 进一步引入编码器-解码器结构优化边界细节。

## 引用
- **原始论文**: [arXiv:1706.05587](https://arxiv.org/abs/1706.05587) | [阅读笔记](../../raw/cnn/DeepLab_v3.md)
