---
id: depthwise_separable_convolution
type: concept
tags: [computer-vision, machine-learning, theoretical]
aliases: [深度可分离卷积, 可分离卷积]
related_nodes: [howard_2017_mobilenet_v1, chollet_2016_xception, convolutional_neural_network]
last_verified: 2026-08-03
---

# Depthwise Separable Convolution

## 定义
将标准卷积分解为逐通道卷积（Depthwise）与逐点卷积（Pointwise 1×1）两步，参数量与计算量降低约 8-9 倍，是高效 CNN 的核心算子。

## 关键要点
- **分解**：先每通道独立卷积，再 1×1 融合通道
- **代价**：表达能力略降，需配合宽度因子补偿
- **地位**：MobileNet/Xception 的基石，也被 Transformer（MHA 分解）借鉴

## 来源
- [[howard_2017_mobilenet_v1]] — MobileNet 核心算子
- [[chollet_2016_xception]] — Xception 极致化应用
