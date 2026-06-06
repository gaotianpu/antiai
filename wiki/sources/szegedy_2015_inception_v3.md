---
id: szegedy_2015_inception_v3
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["Inception v3", "Rethinking the Inception Architecture for Computer Vision", "1512.00567"]
related_nodes: []
arxiv_id: 1512.00567
authors: Christian Szegedy et al.
authors_institution: Google
last_verified: 2026-06-06
---

# Rethinking the Inception Architecture for Computer Vision

- **元数据**: arXiv:1512.00567 | CVPR 2016 | **作者**: Christian Szegedy et al. | **机构**: Google
- **概述**: 系统提出 Inception 架构的设计原则与因式分解卷积技巧
- **关键要点**: 1. 大卷积核因式分解为小卷积核堆叠（如 5×5→2×3×3） 2. 非对称卷积分解（n×n→1×n + n×1） 3. 辅助分类器正则化作用和标签平滑（Label Smoothing）
- **方法/发现**: 通过因式分解和正则化技术，在保持计算效率的同时提升网络深度和精度
- **局限/意义**: 架构设计依然复杂，但设计原则（如因式分解、标签平滑）被广泛采用

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1512.00567) | [阅读笔记](../../raw/cnn/inception_v3.md)
