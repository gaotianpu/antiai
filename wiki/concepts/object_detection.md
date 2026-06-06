---
id: object_detection
type: concept
tags: [machine-learning, empirical-study, survey]
aliases: [目标检测, detection]
related_nodes: [girshick_2013_rcnn, ren_2015_fasterrcnn, redmon_2015_yolov1, redmon_2018_yolov3, ge_2021_yolox, li_2022_yolov6, wang_2022_yolov7, focal_loss_2017, anchor_box]
last_verified: 2026-06-06
---

# Object Detection

## 定义
Object Detection（目标检测）是计算机视觉的核心任务之一：在图像中定位感兴趣物体并识别其类别。输出通常为边界框（bounding box）+ 类别标签 + 置信度。

## 分类范式

### 两阶段检测（Two-Stage）
先提取候选区域（Region Proposal），再对每个候选区域分类和回归。代表：R-CNN（[[girshick_2013_rcnn]]）、Faster R-CNN（[[ren_2015_fasterrcnn]]）、Mask R-CNN（[[mask_rcnn_2017]]）

### 单阶段检测（One-Stage）
直接在特征图上回归类别和边界框，无需候选区域。代表：YOLO 系列（[[redmon_2015_yolov1]]）、SSD、RetinaNet（[[focal_loss_2017]]）

## 关键组件
- **骨干网络** — 特征提取（VGG、ResNet、DarkNet）
- **特征金字塔**（[[fpn_2016]]）— 多尺度特征融合
- **锚框机制**（[[anchor_box]]）— 预定义参考框
- **损失函数** — 分类损失 + 回归损失（如 [[focal_loss_2017]]）

## 来源
- [[girshick_2013_rcnn]] — 基于候选区域的检测奠基
- [[ren_2015_fasterrcnn]] — 端到端两阶段检测
- [[redmon_2015_yolov1]] — 单阶段实时检测
- [[redmon_2018_yolov3]] — YOLOv3 多尺度预测
- [[ge_2021_yolox]] — 无锚框 YOLO
