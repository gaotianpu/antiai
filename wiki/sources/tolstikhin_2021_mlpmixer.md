---
id: tolstikhin_2021_mlpmixer
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["MLP-Mixer", "全MLP视觉架构", "2105.01601"]
related_nodes: []
arxiv_id: 2105.01601
authors: Ilya Tolstikhin et al.
authors_institution: Google
last_verified: 2026-06-06
---

# MLP-Mixer: An All-MLP Architecture for Vision

- **元数据**: arXiv | 2021 | **作者**: Ilya Tolstikhin et al. | **机构**: Google
- **概述**: 提出仅用 MLP 的视觉架构，通过通道混合 + 空间混合 MLP 替代卷积和注意力。
- **关键要点**: 1. 纯 MLP 架构 2. 通道混合 MLP + 空间混合 MLP 3. 性能匹敌 ViT
- **方法/发现**: 证明注意力/卷积不是视觉任务必需的，MLP 足够
- **局限/意义**: 挑战了"注意力机制必要论"，催生后续 MLP-like 架构探索

## 引用
- **原始论文**: [arXiv:2105.01601](https://arxiv.org/abs/2105.01601) | [阅读笔记](../../raw/mlp-mixer.md)
