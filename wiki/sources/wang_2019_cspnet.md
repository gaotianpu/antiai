---
id: wang_2019_cspnet
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["CSPNet", "Cross Stage Partial Network", "1911.11929"]
related_nodes: []
arxiv_id: 1911.11929
authors: Chien-Yao Wang et al.
authors_institution: Academia Sinica
last_verified: 2026-06-06
---

# CSPNet: A New Backbone that can Enhance Learning Capability of CNN

- **元数据**: arXiv:1911.11929 | CVPR 2020 | **作者**: Chien-Yao Wang et al. | **机构**: Academia Sinica
- **概述**: 提出跨阶段局部连接，减少重复梯度计算提升学习能力
- **关键要点**: 1. 将特征图分为两部分，一部分通过主干，另一部分直接拼接 2. 减少重复梯度计算，降低计算量 3. 提出 EFM（Exact Fusion Model）提升多尺度特征融合
- **方法/发现**: 从梯度组合角度解释高基数和稀疏连接的有效性，提出 CSP 设计模式
- **局限/意义**: 对轻量网络改进显著；CSP 设计被 YOLOv4/v5 等检测器广泛采用

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1911.11929) | [阅读笔记](../../raw/cnn/cspnet.md)
