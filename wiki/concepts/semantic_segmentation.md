---
id: semantic_segmentation
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [语义分割, 像素级分类]
related_nodes: [long_2014_fcn, fully_convolutional_network, object_detection]
last_verified: 2026-08-03
---

# Semantic Segmentation

## 定义
对图像每个像素预测语义类别（不区分实例）的视觉任务，是自动驾驶、医学影像、遥感等领域的核心技术。

## 关键要点
- **任务定义**：像素级分类，同类多实例共享标签
- **技术脉络**：FCN → U-Net/SegNet → DeepLab 系列（ASPP）→ PSPNet → Transformer 分割
- **评价指标**：mIoU（平均交并比）

## 来源
- [[long_2014_fcn]] — FCN 开创性工作
