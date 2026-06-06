---
id: han_2019_ghostnet
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["GhostNet", "1911.11907"]
related_nodes: []
arxiv_id: 1911.11907
authors: Kai Han et al.
authors_institution: Huawei
last_verified: 2026-06-06
---

# GhostNet: More Features from Cheap Operations

- **元数据**: arXiv:1911.11907 | CVPR 2020 | **作者**: Kai Han et al. | **机构**: Huawei
- **概述**: 利用特征图冗余，通过廉价线性变换生成 Ghost 特征图
- **关键要点**: 1. Ghost 模块：从内在特征图通过线性变换生成更多特征 2. 即插即用，可升级现有 CNN 3. 同等计算量下优于 MobileNetV3
- **方法/发现**: 利用特征图的冗余性，用廉价操作替代部分标准卷积
- **局限/意义**: 线性变换复杂度对精度有上限约束；但开辟了"利用冗余"的新思路

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1911.11907) | [阅读笔记](../../raw/cnn/GhostNet.md)
