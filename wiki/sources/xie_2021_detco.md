---
id: xie_2021_detco
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["DetCo", "2102.04803"]
related_nodes: []
arxiv_id: 2102.04803
authors: Enze Xie et al.
authors_institution: Nanyang Technological University
last_verified: 2026-06-06
---

# DetCo: Unsupervised Contrastive Learning for Object Detection

- **元数据**: arXiv | 2021 | **作者**: Enze Xie, Jian Ding, Wenhai Wang, Xiaohang Zhan, Hang Xu, Peize Sun, Zhenguo Li, Ping Luo | **机构**: Nanyang Technological University, Huawei Noah's Ark Lab
- **概述**: DetCo 在 MoCo v2 基础上引入多级监督和全局-局部对比学习，同时提升图像分类和目标检测的迁移性能。
- **新颖概念**: —
- **关键要点**: 1. 对 ResNet 各阶段（Res2-Res5）施加多级对比损失 2. 全局图像与局部分块（jigsaw 拼图）交叉对比学习 3. COCO 检测 +0.9 AP over MoCo v2，ImageNet 分类 +1.1%
- **方法/发现**: 多级特征金字塔对比损失 + 全局-局部跨视图对比学习 + 独立记忆库
- **局限/意义**: 协调了分类与检测的矛盾需求，优于同期 DenseCL/InsLoc/PatchReID 等检测友好方法。

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/2102.04803) | [阅读笔记](../../raw/cnn/DetCo.md)
