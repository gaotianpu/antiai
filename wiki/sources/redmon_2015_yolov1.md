---
id: redmon_2015_yolov1
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["YOLO v1", "You Only Look Once", "1506.02640"]
related_nodes: []
arxiv_id: 1506.02640
authors: Joseph Redmon et al.
authors_institution: University of Washington
last_verified: 2026-06-06
---

# You Only Look Once: Unified, Real-Time Object Detection

- **元数据**: arXiv | 2015 | **作者**: Joseph Redmon et al. | **机构**: University of Washington
- **概述**: 提出 YOLO 单阶段目标检测范式，将检测视为回归问题，实现实时端到端检测。
- **关键要点**: 1. 单阶段检测：S×S 网格 + B 个锚框 2. 45 FPS 实时检测 3. 统一框架，速度远快于 R-CNN 系列
- **方法/发现**: 将图像分 S×S 网格，每个网格预测 B 个边界框及其置信度和类别概率
- **局限/意义**: 开创 YOLO 系列，成为工业部署最广泛的目标检测范式

## 引用
- **原始论文**: [arXiv:1506.02640](https://arxiv.org/abs/1506.02640) | [阅读笔记](../../raw/cnn/yolo_v1.md)
