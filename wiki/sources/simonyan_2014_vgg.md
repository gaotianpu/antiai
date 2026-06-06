---
id: simonyan_2014_vgg
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["VGGNet", "Very Deep Convolutional Networks for Large-Scale Image Recognition", "1409.1556"]
related_nodes: []
arxiv_id: 1409.1556
authors: Karen Simonyan et al.
authors_institution: University of Oxford
last_verified: 2026-06-06
---

# Very Deep Convolutional Networks for Large-Scale Image Recognition

- **元数据**: arXiv:1409.1556 | ICLR 2015 | **作者**: Karen Simonyan et al. | **机构**: University of Oxford
- **概述**: 系统研究网络深度对精度的影响，提出 16-19 层 VGG 架构
- **新颖概念**: [[vgg_network]]
- **关键要点**: 1. 全部使用 3×3 小卷积核堆叠 2. 深度增至 16-19 层显著提升性能 3. 在 ImageNet Challenge 2014 定位任务夺冠、分类任务亚军
- **方法/发现**: 证明多个 3×3 卷积层堆叠可等效更大感受野，同时增加非线性
- **局限/意义**: 参数量和计算成本高（仅全连接层就有 122M 参数）；但深度设计思路影响后续 ResNet 等重要工作

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1409.1556) | [阅读笔记](../../raw/cnn/vgg.md)
