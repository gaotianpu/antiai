---
id: anchor_box
type: concept
tags: [machine-learning, empirical-study]
aliases: [锚框, anchor, prior box]
related_nodes: [redmon_2015_yolov1, ren_2015_fasterrcnn, redmon_2016_yolov2, redmon_2018_yolov3]
last_verified: 2026-06-06
---

# Anchor Box

## 定义
Anchor Box（锚框）是一组预定义的边界框模板（不同尺寸和长宽比），用作目标检测中候选检测框的参考基准。模型在每个特征图位置预测相对锚框的偏移量（offset）和置信度，而非直接回归绝对坐标。

## 发展历程
- **Faster R-CNN**（[[ren_2015_fasterrcnn]]）— 首次引入锚框机制
- **YOLOv2**（[[redmon_2016_yolov2]]）— 使用 K-means 聚类从数据中学习最佳锚框尺寸
- **Anchor-Free 趋势** — CornerNet、FCOS 等放弃锚框，直接预测关键点或中心区域

## 来源
- [[ren_2015_fasterrcnn]] — 锚框机制奠基
- [[redmon_2016_yolov2]] — 聚类锚框维度
- [[redmon_2018_yolov3]] — 多尺度锚框预测
