---
id: chen_2020_mocov2
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["MoCo v2", "2003.04297"]
related_nodes: []
arxiv_id: 2003.04297
authors: Xinlei Chen et al.
authors_institution: Facebook AI Research
last_verified: 2026-06-06
---

# Improved Baselines with Momentum Contrastive Learning

- **元数据**: arXiv | 2020 | **作者**: Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He | **机构**: Facebook AI Research
- **概述**: 在 MoCo 框架中集成 SimCLR 的两项改进（MLP 投影头 + 更强数据增广），无需大批量（仅 256）即可超越 SimCLR，称为 MoCo v2。
- **新颖概念**: —
- **关键要点**: 1. MLP 投影头取代线性 fc 头，提升 ImageNet 线性分类 5.6% 2. 额外模糊增广进一步提升检测迁移性能 3. 余弦学习率调度 + 800 epoch 预训练达 71.1%，优于 SimCLR 的 69.3%
- **方法/发现**: MoCo 框架 + MLP 投影头 + 模糊增广 + 余弦学习率
- **局限/意义**: 证明大批量（SimCLR 需 4k-8k）不是对比学习必需，MoCo 框架在 8-GPU 机器即可达到 SOTA。

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/2003.04297) | [阅读笔记](../../raw/cnn/MoCo_v2.md)
