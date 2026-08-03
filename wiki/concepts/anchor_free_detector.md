---
id: anchor_free_detector
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [无锚框检测, Anchor-free]
related_nodes: [ge_2021_yolox, single_stage_detector, object_detection]
last_verified: 2026-08-03
---

# Anchor-Free Detector

## 定义
不预设锚框（anchor），直接在特征图位置上预测目标（中心点/角点/距离）的检测范式，简化设计并消除锚框超参数调优。

## 关键要点
- **关键点检测**：CornerNet（角点）、CenterNet（中心点）
- **中心回归**：FCOS/YOLOX 预测中心到四边距离
- **优势**：少超参数、正样本定义灵活（如 SimOTA 动态分配）

## 来源
- [[ge_2021_yolox]] — YOLOX 转向无锚框
