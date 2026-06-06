---
id: ma_2018_shufflenet_v2
type: source
tags: ["computer-vision", "machine-learning", "practical-guide"]
aliases: ["ShuffleNet v2", "1807.11164"]
related_nodes: []
arxiv_id: 1807.11164
authors: Ningning Ma et al.
authors_institution: Megvii (Face++)
last_verified: 2026-06-06
---

# ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design

- **元数据**: arXiv:1807.11164 | ECCV 2018 | **作者**: Ningning Ma et al. | **机构**: Megvii
- **概述**: 指出 FLOPs 不能准确衡量实际速度，提出 4 条高效网络设计准则
- **关键要点**: 1. FLOPs 非直接速度指标，MAC 和平台特性同样关键 2. 提出 4 条高效网络设计准则 3. 在目标平台上评估直接指标
- **方法/发现**: 通过对照实验揭示 MAC（内存访问成本）对推理速度的关键影响
- **局限/意义**: 设计准则偏向移动端；但纠正了 FLOPs 导向的设计误区

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1807.11164) | [阅读笔记](../../raw/cnn/ShuffleNet_v2.md)
