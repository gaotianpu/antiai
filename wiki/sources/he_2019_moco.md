---
id: he_2019_moco
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["MoCo", "1911.05722"]
related_nodes: []
arxiv_id: 1911.05722
authors: Kaiming He et al.
authors_institution: Facebook AI Research
last_verified: 2026-06-06
---

# Momentum Contrast for Unsupervised Visual Representation Learning

- **元数据**: arXiv | 2019 | **作者**: Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick | **机构**: Facebook AI Research
- **概述**: 提出 MoCo（动量对比），将对比学习视为字典查找，通过队列和动量编码器构建大规模一致字典，缩小无监督与监督表示学习在视觉任务上的差距。
- **关键要点**: 1. 用队列解耦字典大小与小批量大小，支持大量负样本 2. 动量更新（m=0.999）保持键编码器的一致性 3. 在 7 个检测/分割任务上超越监督预训练
- **方法/发现**: InfoNCE 对比损失 + 队列作为动态字典 + 动量编码器 + Shuffling BN
- **局限/意义**: 为后续 MoCo v2/v3 和对比学习系列工作奠定框架基础，表明无监督视觉预训练在多种下游任务中可替代 ImageNet 监督预训练。

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1911.05722) | [阅读笔记](../../raw/cnn/MoCo.md)
