---
id: zhao_2016_pspnet
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["PSPNet", "Pyramid Scene Parsing Network", "1612.01105"]
related_nodes: ["sppnet_2014", "long_2014_fcn", "resnet_2015"]
arxiv_id: 1612.01105
authors: Hengshuang Zhao et al.
authors_institution: CUHK
last_verified: 2026-06-06
---

# Pyramid Scene Parsing Network

- **元数据**: CVPR 2017 | arXiv | 2016 | **作者**: Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia | **机构**: CUHK
- **概述**: 提出金字塔场景解析网络（PSPNet），通过金字塔池化模块在不同尺度上聚合全局上下文信息，解决场景解析中上下文误匹配问题。
- **新颖概念**: [[pyramid_pooling_module]]
- **关键要点**: 1. 金字塔池化模块以 4 种不同尺度（1×1、2×2、3×3、6×6）对特征图进行池化，捕获多尺度上下文 2. 在 PASCAL VOC 2012 达到 85.4% mIoU，Cityscapes 达到 80.2% mIoU 3. 获得 ImageNet 2016 场景解析挑战赛、PASCAL VOC 2012、Cityscapes 三项冠军
- **方法/发现**: 在 FCN 骨干上添加金字塔池化模块，通过全局先验引导像素级预测。
- **局限/意义**: PSPNet 验证了多尺度上下文聚合在密集预测任务中的关键作用，其金字塔池化思想被后续 DeepLab 的 ASPP 和语义分割模型广泛采用。

## 引用
- **原始论文**: [arXiv:1612.01105](https://arxiv.org/abs/1612.01105) | [阅读笔记](../../raw/cnn/pspnet.md)
