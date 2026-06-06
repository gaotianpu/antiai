---
id: zhu_2018_sparsenet
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["SparseNet", "Sparse DenseNet", "1804.05340"]
related_nodes: []
arxiv_id: 1804.05340
authors: Chen Zhu et al.
authors_institution: University of Southern California
last_verified: 2026-06-06
---

# SparseNet: A Sparse DenseNet for Image Classification

- **元数据**: arXiv:1804.05340 | ECCV 2018 | **作者**: Chen Zhu et al. | **机构**: USC
- **概述**: 稀疏化 DenseNet 连接，将连接复杂度从 O(L²) 降至 O(L)
- **新颖概念**: [[sparse_connection]]
- **关键要点**: 1. 将 DenseNet 全连接改为稀疏连接模式 2. 同时增加深度、宽度和连接效率 3. 比 DenseNet 小 2.6 倍、快 3.7 倍
- **方法/发现**: 引入注意力模块进一步提升性能，在 CIFAR/SVHN 上达到 SOTA
- **局限/意义**: 稀疏模式设计需人工指定；但验证了密集连接中大量冗余可被去除

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1804.05340) | [阅读笔记](../../raw/cnn/sparsenet.md)
