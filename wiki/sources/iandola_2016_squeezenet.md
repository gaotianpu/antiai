---
id: iandola_2016_squeezenet
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["SqueezeNet", "1602.07360"]
related_nodes: []
arxiv_id: 1602.07360
authors: Forrest Iandola et al.
authors_institution: UC Berkeley
last_verified: 2026-06-06
---

# SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size

- **元数据**: arXiv:1602.07360 | ICLR 2017 | **作者**: Forrest Iandola et al. | **机构**: UC Berkeley
- **概述**: 在保持 AlexNet 级精度的前提下实现 50 倍参数压缩
- **新颖概念**: [[fire_module]], [[model_pruning]]
- **关键要点**: 1. 提出 Fire module（squeeze + expand 结构） 2. 将 3×3 卷积替换为 1×1 卷积 3. 延迟下采样以保留更多特征信息
- **方法/发现**: 通过 1×1 卷积压缩通道数+混合 1×1/3×3 卷积扩展，大幅减少参数量
- **局限/意义**: 精度稍逊于后续 MobileNet 系列；但开创了轻量化 CNN 设计思路

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1602.07360) | [阅读笔记](../../raw/cnn/SqueezeNet.md)
