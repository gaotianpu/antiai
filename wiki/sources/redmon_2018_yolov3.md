---
id: redmon_2018_yolov3
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["YOLOv3", "An Incremental Improvement", "1804.02767"]
related_nodes: ["redmon_2015_yolov1", "redmon_2016_yolov2"]
arxiv_id: 1804.02767
authors: Joseph Redmon et al.
authors_institution: University of Washington
last_verified: 2026-06-06
---

# YOLOv3: An Incremental Improvement

- **元数据**: arXiv | 2018 | **作者**: Joseph Redmon et al. | **机构**: University of Washington | 相关: [[redmon_2015_yolov1]], [[redmon_2016_yolov2]]
- **概述**: YOLOv3 引入 Darknet-53 骨架 + FPN 多尺度预测，在不牺牲速度的前提下大幅提升小目标检测能力。
- **新颖概念**: [[feature_pyramid]]
- **关键要点**: 1. Darknet-53（ResNet 风格残差骨架） 2. 3 尺度预测（大/中/小目标） 3. 多标签分类（二元交叉熵替代 Softmax）
- **方法/发现**: FPN 风格的多尺度预测显著改善小目标检测
- **局限/意义**: 成为工业界最广泛使用的 YOLO 版本，Joseph Redmon 因伦理担忧停止 CV 研究

## 引用
- **原始论文**: [arXiv:1804.02767](https://arxiv.org/abs/1804.02767) | [阅读笔记](../../raw/cnn/yolo_v3.md)
- **相关概念**: [[redmon_2015_yolov1]], [[redmon_2016_yolov2]]
