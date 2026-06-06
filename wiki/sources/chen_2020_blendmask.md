---
id: chen_2020_blendmask
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["BlendMask", "2001.00309"]
related_nodes: ["mask_rcnn_2017"]
arxiv_id: 2001.00309
authors: Hao Chen et al.
authors_institution: UCSD
last_verified: 2026-06-06
---

# BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation

- **元数据**: AAAI 2020 | arXiv | 2020 | **作者**: Hao Chen, Kunyang Sun, Zhi Tian, Chunhua Shen, Yongming Huang, Youliang Yan | **机构**: UCSD, University of Adelaide, ByteDance
- **概述**: 提出 BlendMask，通过 blender 模块结合自上而下的实例级信息与自下而上的像素级语义信息，在保持单阶段检测框架速度的同时超越 Mask R-CNN 的掩码精度。
- **新颖概念**: —
- **关键要点**: 1. blender 模块以极少通道预测密集的位置敏感实例特征 2. 仅用单卷积层即可学习每个实例的注意力图，推理速度快 3. 轻量版 BlendMask 在 25 FPS 下达到 34.2% mAP，比 Mask R-CNN 快 20%
- **方法/发现**: 将自上而下的实例特征与自下而上的密集语义特征在 blender 模块中融合生成精细掩码。
- **局限/意义**: BlendMask 填补了全卷积实例分割方法与两阶段 Mask R-CNN 之间的精度差距，简单有效的设计使其适合作为实例分割的强基线。

## 引用
- **原始论文**: [arXiv:2001.00309](https://arxiv.org/abs/2001.00309) | [阅读笔记](../../raw/cnn/blendmask.md)
