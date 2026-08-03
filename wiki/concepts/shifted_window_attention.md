---
id: shifted_window_attention
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [移动窗口注意力, SW-MSA]
related_nodes: [swint, self_attention, vision_transformer]
last_verified: 2026-08-03
---

# Shifted Window Attention

## 定义
Swin Transformer 提出的分层注意力：在局部不重叠窗口内计算自注意力，并通过窗口移位使相邻层窗口交叉，实现跨窗口信息流动。

## 关键要点
- **线性复杂度**：窗口大小固定，复杂度随图像尺寸线性增长（对比 ViT 的平方）
- **跨窗口连接**：移位窗口保证不同层间信息交互
- **层次化**：配合 Patch Merging 构建图像金字塔，适配检测/分割

## 来源
- [[swint]] — Swin Transformer 核心设计
