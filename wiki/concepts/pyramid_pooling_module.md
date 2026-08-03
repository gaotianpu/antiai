---
id: pyramid_pooling_module
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [金字塔池化模块, PPM]
related_nodes: [zhao_2016_pspnet, spatial_pyramid_pooling]
last_verified: 2026-08-03
---

# Pyramid Pooling Module (PPM)

## 定义
PSPNet 提出的场景解析模块：在多个尺度（如 1×1/2×2/3×3/6×6）上做全局池化后上采样融合，聚合全局与局部上下文，解决场景误匹配。

## 关键要点
- **多尺度上下文**：不同池化粒度捕获不同范围上下文
- **全局先验**：弥补 FCN 局部感受野导致的类别混淆（如河中小船误判）
- **应用**：场景解析、分割、检测的通用上下文模块

## 来源
- [[zhao_2016_pspnet]] — PSPNet
