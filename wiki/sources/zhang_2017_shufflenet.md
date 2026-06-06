---
id: zhang_2017_shufflenet
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["ShuffleNet", "1707.01083"]
related_nodes: []
arxiv_id: 1707.01083
authors: Xiangyu Zhang et al.
authors_institution: Megvii (Face++)
last_verified: 2026-06-06
---

# ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices

- **元数据**: arXiv:1707.01083 | CVPR 2018 | **作者**: Xiangyu Zhang et al. | **机构**: Megvii
- **概述**: 提出逐点分组卷积 + 通道重排操作，实现极低算力下的高效 CNN
- **新颖概念**: [[channel_shuffle]], [[group_convolution]]
- **关键要点**: 1. 点式分组卷积（pointwise group convolution）降低计算量 2. 通道重排（channel shuffle）促进组间信息流通 3. 40 MFLOPs 下比 MobileNet 绝对提升 7.8% top-1 精度
- **方法/发现**: 通道重排解决了分组卷积中组间信息隔离问题
- **局限/意义**: 分组数需人工设定；但通道重排成为轻量化网络标配

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1707.01083) | [阅读笔记](../../raw/cnn/ShuffleNet.md)
