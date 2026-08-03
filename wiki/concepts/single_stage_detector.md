---
id: single_stage_detector
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [单阶段检测器, 单阶段目标检测]
related_nodes: [redmon_2015_yolov1, object_detection, anchor_free_detector]
last_verified: 2026-08-03
---

# Single-Stage Detector

## 定义
将目标检测视为端到端回归问题、一次前向直接输出类别与框的检测范式（YOLO/SSD），与"先候选后分类"的两阶段范式相对。

## 关键要点
- **速度优势**：无候选阶段，实时检测成为可能
- **精度差距**：早期弱于两阶段（正负样本不平衡），Focal Loss 等弥补
- **演进**：RetinaNet、YOLO 系列、FCOS（无锚框）持续改进

## 来源
- [[redmon_2015_yolov1]] — YOLO 开创单阶段范式
