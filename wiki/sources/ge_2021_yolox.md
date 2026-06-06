---
id: ge_2021_yolox
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["YOLOX", "YOLOX: Exceeding YOLO Series in 2021", "2107.08430"]
related_nodes: ["redmon_2018_yolov3", "bochkovskiy_2020_yolov4"]
arxiv_id: 2107.08430
authors: Zheng Ge et al.
authors_institution: Megvii
last_verified: 2026-06-06
---

# YOLOX: Exceeding YOLO Series in 2021

- **元数据**: arXiv | 2021 | **作者**: Zheng Ge et al. | **机构**: Megvii | 相关: [[redmon_2018_yolov3]], [[bochkovskiy_2020_yolov4]]
- **概述**: 将 YOLO 切换为无锚框检测器，引入解耦头 + SimOTA 标签分配，刷新 COCO SOTA。
- **关键要点**: 1. Anchor-free 设计 2. Decoupled Head 3. SimOTA 标签分配 4. YOLOv3 提升至 47.3% AP
- **方法/发现**: 无锚框设计 + 解耦头显著提升收敛速度和精度
- **局限/意义**: 证明 Anchor-free 在 YOLO 框架中的可行性

## 引用
- **原始论文**: [arXiv:2107.08430](https://arxiv.org/abs/2107.08430) | [阅读笔记](../../raw/cnn/yolox.md)
- **相关概念**: [[redmon_2018_yolov3]], [[bochkovskiy_2020_yolov4]]
