---
id: szegedy_2014_googlenet
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["GoogLeNet", "Inception v1", "Going Deeper with Convolutions", "1409.4842"]
related_nodes: []
arxiv_id: 1409.4842
authors: Christian Szegedy et al.
authors_institution: Google
last_verified: 2026-06-06
---

# Going Deeper with Convolutions

- **元数据**: arXiv:1409.4842 | CVPR 2015 | **作者**: Christian Szegedy et al. | **机构**: Google
- **概述**: 提出 Inception 模块，在增加深度和宽度的同时控制计算复杂度
- **关键要点**: 1. 提出 Inception 模块（多尺度并行卷积 + 1×1 降维） 2. 引入辅助分类器缓解梯度消失 3. ILSVRC 2014 分类任务冠军（top-5 6.67%）
- **方法/发现**: 精心设计局部网络拓扑（Inception 模块）而非简单堆叠层，实现高效计算
- **局限/意义**: 架构设计较为复杂，超参数多；但开创了"网络中的网络"思路，后续 Inception 系列不断演进

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1409.4842) | [阅读笔记](../../raw/cnn/googlenet.md)
