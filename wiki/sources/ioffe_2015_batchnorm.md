---
id: ioffe_2015_batchnorm
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["Batch Normalization", "BN", "1502.03167"]
related_nodes: ["resnet_2015", "szegedy_2015_inception_v3"]
arxiv_id: 1502.03167
authors: Sergey Ioffe et al.
authors_institution: Google
last_verified: 2026-06-06
---

# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

- **元数据**: arXiv | 2015 | **作者**: Sergey Ioffe, Christian Szegedy | **机构**: Google
- **概述**: 提出 Batch Normalization (BN)，通过将每层输入归一化到统一分布来解决内部协变量偏移问题，使训练可以使用更高学习率并减少对 Dropout 的依赖。
- **新颖概念**: [[batch_normalization]]
- **关键要点**: 1. BN 将每个 mini-batch 的激活值归一化为零均值单位方差，并在归一化后引入可学习的缩放与偏移 2. 使网络对初始化不敏感，允许使用 14 倍少的训练步骤达到同等精度 3. 具有正则化效果，在某些情况下可完全替代 Dropout
- **方法/发现**: 在 ImageNet 上将 BN 集成到 Inception 中，达到 4.8% top-5 测试误差，超越人类水平；训练步骤缩减 14 倍。
- **局限/意义**: BN 的 batch 维度依赖在小 batch size 时不稳定，后续被 Layer Norm、Group Norm 等替代方案改进。

## 引用
- **原始论文**: [arXiv:1502.03167](https://arxiv.org/abs/1502.03167) | [阅读笔记](../../raw/cnn/BatchNorm.md)
