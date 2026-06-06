---
id: howard_2017_mobilenet_v1
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["MobileNet v1", "MobileNets", "1704.04861"]
related_nodes: []
arxiv_id: 1704.04861
authors: Andrew Howard et al.
authors_institution: Google
last_verified: 2026-06-06
---

# MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

- **元数据**: arXiv:1704.04861 | CVPR 2017 | **作者**: Andrew Howard et al. | **机构**: Google
- **概述**: 提出深度可分离卷积构建高效移动端 CNN
- **新颖概念**: [[depthwise_separable_convolution]]
- **关键要点**: 1. 深度可分离卷积（depthwise + pointwise）大幅降低计算量 2. 宽度乘数（Width Multiplier）和分辨率乘数 3. 在算力-精度权衡上树立基线
- **方法/发现**: 深度可分离卷积比标准卷积减少 8-9 倍计算量，精度仅有小幅下降
- **局限/意义**: depthwise 卷积实际推理效率受限于底层实现；但奠定了移动端 CNN 的基本范式

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1704.04861) | [阅读笔记](../../raw/cnn/MobileNet_v1.md)
