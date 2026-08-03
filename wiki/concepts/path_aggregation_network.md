---
id: path_aggregation_network
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [路径聚合网络, PANet]
related_nodes: [liu_2018_panet, feature_pyramid, instance_segmentation]
last_verified: 2026-08-03
---

# Path Aggregation Network (PANet)

## 定义
在 FPN 基础上增加自底向上的路径增强与自适应特征池化，改善低层定位信息向高层流动的问题，获 COCO 2017 实例分割冠军。

## 关键要点
- **双向路径**：FPN 自上而下传递语义，PANet 自下而上传递定位
- **自适应池化**：跨层池化融合多级特征供候选使用
- **影响**：PANet 路径成为 YOLOv4/YOLOX 等检测器的标配组件

## 来源
- [[liu_2018_panet]] — PANet
