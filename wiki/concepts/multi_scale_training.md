---
id: multi_scale_training
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [多尺度训练, 尺度增强训练]
related_nodes: [redmon_2016_yolov2, data_augmentation]
last_verified: 2026-08-03
---

# Multi-Scale Training

## 定义
训练中动态改变输入图像分辨率（如每 10 个 batch 随机切换尺度）的训练策略，增强网络对不同尺寸目标的鲁棒性。

## 关键要点
- **实现**：全卷积网络可接受任意尺寸，直接随机缩放输入
- **效果**：提升小目标与多尺度检测性能
- **应用**：YOLOv2 引入，成为检测训练标配

## 来源
- [[redmon_2016_yolov2]] — YOLOv2 多尺度训练
