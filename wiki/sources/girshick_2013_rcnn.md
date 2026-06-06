---
id: girshick_2013_rcnn
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["R-CNN", "Regions with CNN features", "1311.2524"]
related_nodes: []
arxiv_id: 1311.2524
authors: Ross Girshick et al.
authors_institution: UC Berkeley
last_verified: 2026-06-06
---

# Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation

- **元数据**: arXiv | 2013 | **作者**: Ross Girshick et al. | **机构**: UC Berkeley
- **概述**: 提出 R-CNN，将 CNN 与区域候选结合，在 VOC 2012 上 mAP 提升超 30%。
- **关键要点**: 1. 区域候选（Selective Search）+ CNN 分类 2. 迁移学习（ImageNet 预训练 + 微调） 3. VOC 2012 mAP 53.3%
- **方法/发现**: CNN 特征提取 + SVM 分类 + Bounding Box 回归三阶段
- **局限/意义**: 开创深度学习的目标准检测范式，后续 Faster R-CNN/Mask R-CNN 的基石

## 引用
- **原始论文**: [arXiv:1311.2524](https://arxiv.org/abs/1311.2524) | [阅读笔记](../../raw/cnn/R-CNN.md)
