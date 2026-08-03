---
id: region_proposal
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [区域候选, 候选区域]
related_nodes: [girshick_2013_rcnn, region_proposal_network, object_detection]
last_verified: 2026-08-03
---

# Region Proposal

## 定义
目标检测两阶段范式的第一步：从图像中生成可能包含目标的候选区域（如选择性搜索约 2000 个），再逐个分类。

## 关键要点
- **R-CNN 范式**：选择性搜索生成候选 → CNN 特征 → SVM/回归
- **瓶颈**：候选区域多、逐区域前向计算昂贵
- **演进**：RPN 将候选生成也网络化，实现端到端

## 来源
- [[girshick_2013_rcnn]] — R-CNN
