---
id: wang_2017_nonlocal
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["Non-local Neural Networks", "Non-local", "1711.07971"]
related_nodes: ["resnet_2015", "mask_rcnn_2017"]
arxiv_id: 1711.07971
authors: Xiaolong Wang et al.
authors_institution: CMU
last_verified: 2026-06-06
---

# Non-local Neural Networks

- **元数据**: CVPR 2018 | arXiv | 2017 | **作者**: Xiaolong Wang, Ross Girshick, Abhinav Gupta, Kaiming He | **机构**: CMU, Facebook AI Research
- **概述**: 提出非局部（Non-local）操作作为通用构建块来捕获长距离依赖，在视频分类和图像检测/分割任务上取得显著提升。
- **关键要点**: 1. 受经典非局部均值去噪启发，将每个位置的响应计算为所有位置特征的加权和 2. 可嵌入任意 CNN 架构，在视频分类中无需额外技巧即可超越竞赛冠军 3. 在 COCO 上改进了目标检测/分割和姿态估计
- **方法/发现**: 在 ResNet 骨架中插入非局部块，在 Kinetics 和 Charades 上取得 SOTA 结果。
- **局限/意义**: Non-local 的计算复杂度为 $O(HW \times HW)$，在高分辨率输入上开销大，后续被更高效的注意力机制（如 Self-attention 简化版）改进。

## 引用
- **原始论文**: [arXiv:1711.07971](https://arxiv.org/abs/1711.07971) | [阅读笔记](../../raw/cnn/Non-local.md)
