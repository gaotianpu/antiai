---
id: region_proposal_network
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [区域候选网络, RPN]
related_nodes: [ren_2015_fasterrcnn, region_proposal, anchor_box, object_detection]
last_verified: 2026-08-03
---

# Region Proposal Network (RPN)

## 定义
Faster R-CNN 提出的可学习候选生成网络：在共享特征图上滑动预测"锚框是否含目标 + 边框回归"，将候选生成融入检测网络。

## 关键要点
- **端到端**：候选生成与检测联合训练，替代选择性搜索
- **锚框机制**：每个位置多尺度多长宽比锚框
- **速度**：检测从秒级降到近实时（~5-17 fps 时代）

## 来源
- [[ren_2015_fasterrcnn]] — Faster R-CNN
