---
id: chen_2020_simclr
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["SimCLR", "2002.05709"]
related_nodes: []
arxiv_id: 2002.05709
authors: Ting Chen et al.
authors_institution: Google Research
last_verified: 2026-06-06
---

# A Simple Framework for Contrastive Learning of Visual Representations

- **元数据**: arXiv | 2020 | **作者**: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton | **机构**: Google Research
- **概述**: 提出 SimCLR，一个简洁的对比学习框架，无需专门架构或记忆库，通过数据增广组合 + MLP 投影头 + 大批量训练实现 SOTA。
- **关键要点**: 1. 数据增广组合（裁剪+颜色失真+模糊）对定义有效预测任务至关重要 2. MLP 投影头大幅提升表示质量 3. 大批量（4096-8192）和长训练提供更多负样本
- **方法/发现**: 端到端对比学习 + NT-Xent 损失 + 全局 BN + LARS 优化器
- **局限/意义**: 线性分类达 76.5% Top-1，匹配监督 ResNet-50；但需大批量依赖 TPU，计算开销大。

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/2002.05709) | [阅读笔记](../../raw/cnn/SimCLR.md)
