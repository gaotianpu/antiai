---
id: feature_pyramid
type: concept
tags: [machine-learning, theoretical]
aliases: [特征金字塔, FPN, 多尺度特征]
related_nodes: [convolutional_neural_network, object_detection]
---

# Feature Pyramid

## 概述

Feature Pyramid（特征金字塔，FPN）是一种多尺度特征表示架构，通过在卷积网络中构建自顶向下的侧连接，使每一层都融合高语义的深层特征和高分辨率的浅层特征，显著提升目标检测和分割任务在尺度变化场景下的表现。

## 详细阐述

### 定义与内涵

传统的图像金字塔（Image Pyramid）对图像缩放后独立计算特征，计算量巨大。FPN 利用 CNN 本身的金字塔层级结构，通过**自底向上通路 + 自顶向下通路 + 横向连接**构建特征金字塔，在不显著增加计算量的前提下为每层赋予丰富的语义信息。

### 关键架构

| 架构 | 核心改进 | 出处 |
|:---|:---|:---|
| FPN | 横向连接 + 自上而下上采样融合 | [[fpn_2016]] |
| PANet | 在 FPN 基础上增加自底向上的额外路径（PA） | [[liu_2018_panet]] |
| NAS-FPN | 神经架构搜索自动设计金字塔结构 | — |
| BiFPN | 加权特征融合 + 双向跨尺度连接（EfficientDet） | [[tan_2019_efficientnet]] 同系列 |
| YOLOv4 中的 PANet | 结合 SPP 的 PANet 变体 | [[bochkovskiy_2020_yolov4]] |

### 设计要素

- **横向连接**：通常是 1×1 卷积，将自底向上特征图通道对齐
- **融合方式**：逐元素相加（FPN）或连接后卷积（PANet）
- **输出尺度**：多尺度输出用于分别检测不同尺寸目标

### 应用场景

- 目标检测：[[object_detection]] 中的标准 Neck 组件
- 实例分割：Mask R-CNN
- 语义分割：[[chen_2017_deeplabv3]] 等多尺度融合
- 姿态估计：多尺度特征提升关键点检测精度

## 相关概念网络

- [[convolutional_neural_network]] — FPN 构建在 CNN 特征层级之上
- [[object_detection]] — FPN 是检测器中多尺度处理的标配
- [[residual_connection]] — 类似的自上而下连接思想

## 引用资料

1. [[fpn_2016]] — 原始 FPN 论文
2. [[liu_2018_panet]] — PANet 路径聚合增强
3. [[bochkovskiy_2020_yolov4]] — YOLOv4 中的 FPN/PANet 实践

## 更新记录

- **2026-06-06**: 初始创建
