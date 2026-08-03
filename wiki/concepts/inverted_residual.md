---
id: inverted_residual
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [倒残差, 逆残差结构]
related_nodes: [sandler_2018_mobilenet_v2, linear_bottleneck, residual_connection]
last_verified: 2026-08-03
---

# Inverted Residual

## 定义
MobileNetV2 提出的残差变体：先升维（1×1 扩展）→ depthwise 卷积 → 降维（1×1 压缩），与 ResNet"降维-升维"的瓶颈方向相反。

## 关键要点
- **反直觉设计**：低维输入直接进 depthwise 会丢失信息，先扩展再卷积
- **短接在低维**：残差连接位于压缩后的瓶颈层
- **效果**：同等算力下精度显著优于 v1

## 来源
- [[sandler_2018_mobilenet_v2]] — MobileNetV2
