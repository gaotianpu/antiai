---
id: instance_segmentation
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [实例分割]
related_nodes: [mask_rcnn_2017, semantic_segmentation, object_detection]
last_verified: 2026-08-03
---

# Instance Segmentation

## 定义
同时完成检测（定位实例）与分割（像素级掩码）的任务：既要区分不同实例，又要给出每个实例的精确轮廓。

## 关键要点
- **任务组合**：检测 + 语义分割的精细化
- **两阶段范式**：Mask R-CNN（检测分支 + 掩码分支）为经典基线
- **单阶段范式**：YOLACT、BlendMask 等追求实时性

## 来源
- [[mask_rcnn_2017]] — Mask R-CNN
