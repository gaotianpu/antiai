---
id: compound_scaling
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [复合缩放, 复合扩展]
related_nodes: [tan_2019_efficientnet, neural_architecture_search]
last_verified: 2026-08-03
---

# Compound Scaling

## 定义
EfficientNet 提出的模型缩放方法：用系数 φ 统一缩放深度、宽度、分辨率三个维度（$d=\alpha^\phi, w=\beta^\phi, r=\gamma^\phi$），比单一维度缩放更高效。

## 关键要点
- **协同缩放**：分辨率提高需要更深更宽的网络配合
- **NAS 基线**：先用 NAS 搜索基线网络，再复合缩放
- **成果**：EfficientNet 以更少 FLOPs 刷新 ImageNet 精度纪录

## 来源
- [[tan_2019_efficientnet]] — EfficientNet
