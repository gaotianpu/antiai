---
id: atrous_spatial_pyramid_pooling
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [空洞空间金字塔池化, ASPP]
related_nodes: [chen_2017_deeplabv3, atrous_convolution, spatial_pyramid_pooling]
last_verified: 2026-08-03
---

# Atrous Spatial Pyramid Pooling (ASPP)

## 定义
DeepLab 提出的多尺度上下文聚合模块：用多个不同膨胀率的空洞卷积并行提取特征并融合，捕获多尺度目标上下文。

## 关键要点
- **多尺度并行**：不同膨胀率对应不同感受野
- **图像级特征**：v3 加入全局平均池化分支补充全局上下文
- **地位**：语义分割 SOTA 架构（DeepLab 系列）的核心组件

## 来源
- [[chen_2017_deeplabv3]] — DeepLabv3 改进 ASPP
