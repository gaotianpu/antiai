---
id: transfuser
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [TransFuser, 跨模态驾驶融合]
related_nodes: [prakash_2021_transfuser, multimodal_fusion, end_to_end_driving]
last_verified: 2026-08-03
---

# TransFuser

## 定义
自动驾驶的端到端多模态架构：用 Transformer 注意力机制融合相机图像与 LiDAR 鸟瞰特征，生成驾驶指令，在 CARLA 基准上显著超越基线。

## 关键要点
- **互补模态**：相机（语义细节）+ LiDAR（几何/深度）
- **注意力融合**：跨模态注意力模块自适应整合两路特征
- **意义**：证明注意力融合优于简单拼接/平均

## 来源
- [[prakash_2021_transfuser]] — TransFuser
