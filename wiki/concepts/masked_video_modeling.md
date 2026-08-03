---
id: masked_video_modeling
type: concept
tags: [machine-learning, computer-vision, empirical-study]
aliases: [掩码视频建模, 视频掩码预训练]
related_nodes: [tong_2022_maest, masked_image_modeling]
last_verified: 2026-08-03
---

# Masked Video Modeling

## 定义
将 MIM 扩展到视频：掩码视频块（空间+时间）并重建，如 MAE-ST 在高掩码率下学习时空表示。

## 关键要点
- **时空掩码**：同时掩码空间块与时间片段，掩码率可更高（90%+）
- **任务差异**：重建目标可为像素、特征或 token
- **效果**：视频分类/检测等下游任务的强预训练范式

## 来源
- [[tong_2022_maest]] — MAE-ST
