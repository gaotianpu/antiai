---
id: atrous_convolution
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [空洞卷积, 膨胀卷积, Atrous Conv]
related_nodes: [chen_2017_deeplabv3, atrous_spatial_pyramid_pooling]
last_verified: 2026-08-03
---

# Atrous Convolution

## 定义
在卷积核元素间插入空洞（膨胀率 r）扩大感受野的卷积，不增加参数与计算量，同时保持特征图分辨率，是语义分割的关键算子。

## 关键要点
- **感受野控制**：膨胀率 r 使 3×3 核感受野扩至 (2r+1)²
- **分辨率保持**：对比池化+下采样，不损失空间细节
- **应用**：DeepLab 系列、RFB 模块等

## 来源
- [[chen_2017_deeplabv3]] — DeepLabv3 系统研究空洞卷积
