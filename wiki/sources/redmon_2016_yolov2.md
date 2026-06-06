---
id: redmon_2016_yolov2
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["YOLOv2", "YOLO9000", "1612.08242"]
related_nodes: ["redmon_2015_yolov1"]
arxiv_id: 1612.08242
authors: Joseph Redmon et al.
authors_institution: University of Washington
last_verified: 2026-06-06
---

# YOLO9000: Better, Faster, Stronger

- **元数据**: arXiv | 2016 | **作者**: Joseph Redmon et al. | **机构**: University of Washington | 相关: [[redmon_2015_yolov1]]
- **概述**: YOLOv2 在 v1 基础上引入批量归一化、锚框先验、多尺度训练，YOLO9000 联合检测和分类数据集实现 9000 类检测。
- **关键要点**: 1. Darknet-19 骨架 2. 锚框先验聚类 3. 多尺度训练 4. WordTree 联合 9000 类检测
- **方法/发现**: 通过分类和检测数据联合训练实现零样本检测泛化
- **局限/意义**: 多尺度训练成为后续检测模型标配

## 引用
- **原始论文**: [arXiv:1612.08242](https://arxiv.org/abs/1612.08242) | [阅读笔记](../../raw/cnn/yolo_v2.md)
- **相关概念**: [[redmon_2015_yolov1]]
