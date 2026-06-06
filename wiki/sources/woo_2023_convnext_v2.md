---
id: woo_2023_convnext_v2
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["ConvNeXt V2", "2301.00808"]
related_nodes: []
arxiv_id: 2301.00808
authors: Sanghyun Woo et al.
authors_institution: Facebook AI Research
last_verified: 2026-06-06
---

# ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders

- **元数据**: arXiv:2301.00808 | CVPR 2023 | **作者**: Sanghyun Woo et al. | **机构**: Facebook AI Research
- **概述**: 将 ConvNeXt 与掩码自编码器（MAE）协同设计，提升自监督学习性能
- **新颖概念**: —
- **关键要点**: 1. 全卷积掩码自编码器（FCMAE）框架 2. 提出全局响应归一化（GRN）解决特征塌陷 3. 650M Huge 模型达 88.9% ImageNet top-1 精度
- **方法/发现**: 发现 MAE + ConvNeXt 直接结合效果不佳，GRN 解决 MLP 层特征塌陷
- **局限/意义**: 依赖 MAE 预训练框架；但证明了自监督与 CNN 架构协同设计的价值

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/2301.00808) | [阅读笔记](../../raw/cnn/ConvNeXt_v2.md)
