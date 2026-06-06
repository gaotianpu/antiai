---
id: gao_2019_res2net
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["Res2Net", "1904.01169"]
related_nodes: ["resnet_2015", "xie_2016_resnext"]
arxiv_id: 1904.01169
authors: Shang-Hua Gao et al.
authors_institution: Nankai University
last_verified: 2026-06-06
---

# Res2Net: A New Multi-scale Backbone Architecture

- **元数据**: IEEE TPAMI 2021 | arXiv | 2019 | **作者**: Shang-Hua Gao, Ming-Ming Cheng, Kai Zhao, Xin-Yu Zhang, Ming-Hsuan Yang, Philip Torr | **机构**: Nankai University, UC Merced, Oxford University
- **概述**: 提出 Res2Net 模块，通过在单个残差块内构建层次化残差连接来实现细粒度的多尺度特征表示。
- **新颖概念**: [[multi_scale_representation]]
- **关键要点**: 1. 将单个残差块内的 3×3 卷积替换为一组更小的卷积组，以层次化残差方式连接 2. 在不显著增加计算量的前提下扩大了每层的感受野范围 3. 可插入 ResNet、ResNeXt、DLA 等主流骨干实现性能提升
- **方法/发现**: 在 CIFAR-100、ImageNet 分类以及目标检测、显著性检测等下游任务中验证了持续一致的性能提升。
- **局限/意义**: Res2Net 的层级分组设计增加了超参数（分组数、宽度），需要针对不同规模模型进行调优。

## 引用
- **原始论文**: [arXiv:1904.01169](https://arxiv.org/abs/1904.01169) | [阅读笔记](../../raw/cnn/Res2Net.md)
