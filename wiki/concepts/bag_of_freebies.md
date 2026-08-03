---
id: bag_of_freebies
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [免费赠品集, 训练技巧包]
related_nodes: [bochkovskiy_2020_yolov4, trainable_bag_of_freebies]
last_verified: 2026-08-03
---

# Bag of Freebies

## 定义
YOLOv4 术语：只改变训练策略、不增加推理成本的技巧集合（数据增强、损失设计、正则化等），如 Mosaic、CIoU、Mish、标签平滑。

## 关键要点
- **零推理代价**：技巧只在训练期生效
- **系统评估**：YOLOv4 逐一验证各技巧对检测器的贡献
- **分类**：与"Bag of Specials"（增加少量推理成本换精度）相对

## 来源
- [[bochkovskiy_2020_yolov4]] — YOLOv4 技巧系统梳理
