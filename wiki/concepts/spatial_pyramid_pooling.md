---
id: spatial_pyramid_pooling
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [空间金字塔池化, SPP]
related_nodes: [sppnet_2014, pyramid_pooling_module, fully_convolutional_network]
last_verified: 2026-08-03
---

# Spatial Pyramid Pooling (SPP)

## 定义
将特征图划分为多个尺度的网格分别池化后拼接为固定长度向量，使网络接受任意尺寸输入，并聚合多尺度信息。

## 关键要点
- **固定输出**：无论输入尺寸，输出维度一致，替代全连接前的全局池化
- **多尺度感受野**：不同网格粒度对应不同区域粒度
- **影响**：催生 SPPNet、PSPNet、ASPP 等系列工作

## 来源
- [[sppnet_2014]] — SPPNet
