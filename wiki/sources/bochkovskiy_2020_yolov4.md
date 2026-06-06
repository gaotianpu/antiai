---
id: bochkovskiy_2020_yolov4
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["YOLOv4", "Optimal Speed and Accuracy", "2004.10934"]
related_nodes: ["redmon_2018_yolov3"]
arxiv_id: 2004.10934
authors: Alexey Bochkovskiy et al.
authors_institution: Academia Sinica
last_verified: 2026-06-06
---

# YOLOv4: Optimal Speed and Accuracy of Object Detection

- **元数据**: arXiv | 2020 | **作者**: Alexey Bochkovskiy et al. | **机构**: Academia Sinica | 相关: [[redmon_2018_yolov3]]
- **概述**: 系统性梳理检测器各组件的影响，组合 CSPDarknet-53 + PANet + MISH + CIoU 等技巧实现 SOTA。
- **关键要点**: 1. CSPDarknet-53 骨架 2. PANet 特征聚合 3. MISH 激活 + CIoU 损失 4. Bag-of-Freebies 系统消融
- **方法/发现**: 对 Backbone/Neck/Head/Training 的每项改进做系统消融实验
- **局限/意义**: 成为"训练技巧百科全书"，后续 YOLO 变体的参照基线

## 引用
- **原始论文**: [arXiv:2004.10934](https://arxiv.org/abs/2004.10934) | [阅读笔记](../../raw/cnn/yolo_v4.md)
- **相关概念**: [[redmon_2018_yolov3]]
