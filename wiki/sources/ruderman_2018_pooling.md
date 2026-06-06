---
id: ruderman_2018_pooling
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["Pooling Deformation Stability", "1804.04438"]
related_nodes: ["simonyan_2014_vgg", "krizhevsky_2012_alexnet"]
arxiv_id: 1804.04438
authors: Avraham Ruderman et al.
authors_institution: DeepMind
last_verified: 2026-06-06
---

# Pooling is neither necessary nor sufficient for appropriate deformation stability in CNNs

- **元数据**: NIPS 2018 | arXiv | 2018 | **作者**: Avraham Ruderman, Neil C. Rabinowitz, Ari S. Morcos, Daniel Zoran | **机构**: DeepMind
- **概述**: 通过系统实验推翻"池化层是 CNN 变形稳定性必要条件"的假设，发现池化既非必要也非充分，且网络在训练中会主动抵消池化的过度稳定性偏置。
- **新颖概念**: —
- **关键要点**: 1. 变形不变性不是二元属性，不同任务在不同层需要不同程度稳定性 2. 变形稳定性在训练中动态调整，主要通过卷积滤波器平滑度实现 3. 池化在初始化时带来过多变形稳定性，网络需学会抵消这一归纳偏置
- **方法/发现**: 使用控制变量法在 ImageNet 分类任务上分离池化层与变形稳定性的因果关系。
- **局限/意义**: 挑战了 CNN 设计的经典直觉，解释了现代 CNN 减少池化层但仍保持变形鲁棒性的原因。

## 引用
- **原始论文**: [arXiv:1804.04438](https://arxiv.org/abs/1804.04438) | [阅读笔记](../../raw/cnn/Pooling.md)
