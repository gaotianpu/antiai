---
id: ding_2021_repvgg
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["RepVGG", "2101.03697"]
related_nodes: ["simonyan_2014_vgg", "resnet_2015", "radosavovic_2020_regnet", "tan_2019_efficientnet"]
arxiv_id: 2101.03697
authors: Xiaohan Ding et al.
authors_institution: Tsinghua University
last_verified: 2026-06-06
---

# RepVGG: Making VGG-style ConvNets Great Again

- **元数据**: CVPR 2021 | arXiv | 2021 | **作者**: Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, Jian Sun | **机构**: Tsinghua University, Megvii Research
- **概述**: 提出 RepVGG，通过结构重参数化技术在训练时使用多分支拓扑、推理时合并为纯 3×3 卷积堆叠的 VGG 风格单路径网络，实现速度与精度的最佳权衡。
- **关键要点**: 1. 训练时使用 3×3 conv + 1×1 conv + identity 的多分支结构，推理时通过重参数化合并为单路 3×3 conv 2. 首次使用普通架构在 ImageNet 上达到超过 80% top-1 精度 3. 比 ResNet-50 快 83%、比 ResNet-101 快 101%，精度更高
- **方法/发现**: 结构重参数化（Structural Re-parameterization）技术将训练时复杂拓扑等价转换为推理时简单结构。
- **局限/意义**: 重参数化技术启发了后续 RepMLP、RepOptimizer 等工作，成为模型加速的重要范式。但训练时多分支结构占用更多显存。

## 引用
- **原始论文**: [arXiv:2101.03697](https://arxiv.org/abs/2101.03697) | [阅读笔记](../../raw/cnn/repvgg.md)
