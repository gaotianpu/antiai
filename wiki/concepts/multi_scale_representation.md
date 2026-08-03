---
id: multi_scale_representation
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [多尺度表示, 多尺度特征]
related_nodes: [gao_2019_res2net, feature_pyramid, convolutional_neural_network]
last_verified: 2026-08-03
---

# Multi-Scale Representation

## 定义
在同一网络/模块内同时建模多种尺度特征的表示方式，提升对不同尺寸目标的感知能力。

## 关键要点
- **Res2Net 方案**：在单个残差块内用多个子感受野分支构建层次化多尺度
- **宏观方案**：FPN 用特征金字塔、PSPNet 用多尺度池化
- **价值**：小目标检测、场景解析等任务的共性需求

## 来源
- [[gao_2019_res2net]] — Res2Net 细粒度多尺度
