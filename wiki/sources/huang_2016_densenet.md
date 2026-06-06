---
id: huang_2016_densenet
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["DenseNet", "Densely Connected Convolutional Networks", "1608.06993"]
related_nodes: []
arxiv_id: 1608.06993
authors: Gao Huang et al.
authors_institution: Cornell University
last_verified: 2026-06-06
---

# Densely Connected Convolutional Networks

- **元数据**: arXiv:1608.06993 | CVPR 2017 | **作者**: Gao Huang et al. | **机构**: Cornell University
- **概述**: 提出密集连接机制，每层与之前所有层直接连接
- **关键要点**: 1. L 层网络有 L(L+1)/2 个直接连接 2. 缓解梯度消失、加强特征传播与复用 3. 大幅减少参数量（无需学习冗余特征）
- **方法/发现**: 通过特征图在通道维度的拼接（而非相加）实现密集连接，提升信息流
- **局限/意义**: 显存占用高（需存储大量中间特征图）；但特征复用思想影响 HarDNet 等工作

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1608.06993) | [阅读笔记](../../raw/cnn/densenet.md)
