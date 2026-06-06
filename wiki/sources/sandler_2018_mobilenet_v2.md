---
id: sandler_2018_mobilenet_v2
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["MobileNet v2", "MobileNetV2", "1801.04381"]
related_nodes: []
arxiv_id: 1801.04381
authors: Mark Sandler et al.
authors_institution: Google
last_verified: 2026-06-06
---

# MobileNetV2: Inverted Residuals and Linear Bottlenecks

- **元数据**: arXiv:1801.04381 | CVPR 2018 | **作者**: Mark Sandler et al. | **机构**: Google
- **概述**: 提出倒残差结构（Inverted Residual）和线性瓶颈（Linear Bottleneck）
- **新颖概念**: [[inverted_residual]], [[linear_bottleneck]]
- **关键要点**: 1. 倒残差：在瓶颈层间跳跃连接，中间扩展高维 2. 线性瓶颈：窄层去除非线性保持表征能力 3. 提出 SSDLite 轻量检测框架
- **方法/发现**: 倒残差提升移动端模型精度，线性瓶颈避免信息丢失
- **局限/意义**: 架构设计精巧但复杂度增加；但倒残差成为 MobileNet 系列的核心设计

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1801.04381) | [阅读笔记](../../raw/cnn/MobileNet_v2.md)
