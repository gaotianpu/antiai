---
id: long_2014_fcn
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["FCN", "Fully Convolutional Networks", "1411.4038"]
related_nodes: ["simonyan_2014_vgg", "szegedy_2014_googlenet", "sppnet_2014"]
arxiv_id: 1411.4038
authors: Jonathan Long et al.
authors_institution: UC Berkeley
last_verified: 2026-06-06
---

# Fully Convolutional Networks for Semantic Segmentation

- **元数据**: CVPR 2015 | arXiv | 2014 | **作者**: Jonathan Long, Evan Shelhamer, Trevor Darrell | **机构**: UC Berkeley
- **概述**: 提出全卷积网络（FCN），将分类网络中的全连接层替换为卷积层，实现端到端、像素到像素的语义分割，接受任意尺寸输入并产生对应尺寸输出。
- **新颖概念**: [[fully_convolutional_network]], [[semantic_segmentation]]
- **关键要点**: 1. 将 AlexNet、VGG、GoogLeNet 改造为全卷积架构并通过微调迁移至分割任务 2. 跳跃连接（skip connection）将深层语义信息与浅层外观信息融合以生成精细分割图 3. 推理时间不到 0.2 秒/图，在 PASCAL VOC 2012 上以 62.2% mIoU 超越 SOTA 20%
- **方法/发现**: 反卷积层（转置卷积）实现上采样，skip 连接恢复空间分辨率。
- **局限/意义**: FCN 奠定了深度学习语义分割的基础范式。后续 U-Net、DeepLab、PSPNet 等均在此基础上改进，主要局限在于大感受野与空间细节的权衡。

## 引用
- **原始论文**: [arXiv:1411.4038](https://arxiv.org/abs/1411.4038) | [阅读笔记](../../raw/cnn/FCN.md)
