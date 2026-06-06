---
id: ding_2021_repmlp
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["RepMLP", "卷積重参数化MLP", "2105.01883"]
related_nodes: ["tolstikhin_2021_mlpmixer"]
arxiv_id: 2105.01883
authors: Xiaohan Ding et al.
authors_institution: Tsinghua University
last_verified: 2026-06-06
---

# RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition

- **元数据**: arXiv | 2021 | **作者**: Xiaohan Ding et al. | **机构**: Tsinghua University | 相关: [[tolstikhin_2021_mlpmixer]]
- **概述**: 将卷积重参数化为全连接层，在训练时使用卷积结构，推理时等效为 MLP。
- **关键要点**: 1. 训练时卷积 → 推理时 MLP 2. 重参数化 3. 兼顾 CNN 的局部先验和 MLP 的高效推理
- **方法/发现**: 结构重参数化技术，训练和推理使用不同等价结构
- **局限/意义**: 将重参数化思想引入 MLP-like 架构

## 引用
- **原始论文**: [arXiv:2105.01883](https://arxiv.org/abs/2105.01883) | [阅读笔记](../../raw/repmlp.md)
- **相关概念**: [[tolstikhin_2021_mlpmixer]]
