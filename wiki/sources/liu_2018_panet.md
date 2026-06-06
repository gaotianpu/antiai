---
id: liu_2018_panet
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["PANet", "Path Aggregation Network", "1803.01534"]
related_nodes: ["fpn_2016", "mask_rcnn_2017"]
arxiv_id: 1803.01534
authors: Shu Liu et al.
authors_institution: HKUST
last_verified: 2026-06-06
---

# Path Aggregation Network for Instance Segmentation

- **元数据**: CVPR 2018 | arXiv | 2018 | **作者**: Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, Jiaya Jia | **机构**: HKUST, CUHK
- **概述**: 提出路径聚合网络（PANet），通过自底向上的路径增强和自适应特征池化来改进基于提议的实例分割框架中的信息流动，获得 COCO 2017 实例分割冠军。
- **新颖概念**: [[path_aggregation_network]]
- **关键要点**: 1. 自底向上路径增强缩短特征金字塔中低层与顶层之间的信息路径 2. 自适应特征池化将特征网格与所有特征层级连接，使有用信息直接传播到后续子网络 3. 为每个提议创建补充分支捕获不同视图以改善掩码预测
- **方法/发现**: 在 Mask R-CNN 框架基础上增加自底向上增强路径和自适应特征池化。
- **局限/意义**: PANet 是 FPN 之后特征聚合的重要改进，其自适应池化思想后被更简化的方法（如 YOLOF）改进。

## 引用
- **原始论文**: [arXiv:1803.01534](https://arxiv.org/abs/1803.01534) | [阅读笔记](../../raw/cnn/panet.md)
