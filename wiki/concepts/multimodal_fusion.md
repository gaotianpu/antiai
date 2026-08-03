---
id: multimodal_fusion
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [多模态融合, 跨模态融合]
related_nodes: [prakash_2021_transfuser, multimodal_language_model]
last_verified: 2026-08-03
---

# Multimodal Fusion

## 定义
融合多种传感器/模态（相机、LiDAR、文本等）信息的机制：早期/中期/晚期融合策略，以及基于注意力的自适应融合。

## 关键要点
- **融合层级**：输入级（early）、特征级（mid）、决策级（late）
- **注意力融合**：TransFuser 用 Transformer 注意力融合图像与 LiDAR 特征
- **意义**：自动驾驶、多模态理解的核心问题——互补信息如何结合

## 来源
- [[prakash_2021_transfuser]] — TransFuser 注意力融合
