---
id: liu_2022_convnext
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["ConvNeXt", "A ConvNet for the 2020s", "2201.03545"]
related_nodes: []
arxiv_id: 2201.03545
authors: Zhuang Liu et al.
authors_institution: Facebook AI Research
last_verified: 2026-06-06
---

# A ConvNet for the 2020s

- **元数据**: arXiv:2201.03545 | CVPR 2022 | **作者**: Zhuang Liu et al. | **机构**: Facebook AI Research
- **概述**: 借鉴 Swin Transformer 设计策略，现代化纯 CNN 架构
- **新颖概念**: —
- **关键要点**: 1. 从 ResNet 出发，逐步引入 ViT 设计元素（patchify、LN、GELU 等） 2. 大卷积核（7×7）深度可分离卷积 3. 纯 CNN 架构达到与 Swin Transformer 相当的性能
- **方法/发现**: 通过可控消融实验，溯源 ViT 成功的关键设计元素并迁移到 CNN
- **局限/意义**: 架构设计受 ViT 启发，原创性有限；但证明了纯 CNN 通过现代化设计可媲美 Transformer

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/2201.03545) | [阅读笔记](../../raw/cnn/ConvNeXt.md)
