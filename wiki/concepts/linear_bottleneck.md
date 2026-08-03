---
id: linear_bottleneck
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [线性瓶颈, Linear Bottleneck]
related_nodes: [sandler_2018_mobilenet_v2, inverted_residual]
last_verified: 2026-08-03
---

# Linear Bottleneck

## 定义
MobileNetV2 的设计原则：瓶颈层（低维）不接 ReLU 非线性，保持线性变换，避免低维空间中非线性造成的特征破坏。

## 关键要点
- **动机**：ReLU 在低维流形上会大量置零导致信息丢失
- **实现**：瓶颈层输出前去掉激活（或使用线性激活）
- **配合**：与倒残差结构共同构成 MobileNetV2 的核心

## 来源
- [[sandler_2018_mobilenet_v2]] — MobileNetV2 线性瓶颈
