---
id: instance_segmentation_survey
type: source
tags: ["computer-vision", "machine-learning", "survey"]
aliases: ["实例分割综述", "Instance Segmentation"]
related_nodes: ["mask_rcnn_2017", "fpn_2016"]
arxiv_id: null
authors: null
authors_institution: null
last_verified: 2026-06-06
---

# 实例分割 Instance Segmentation

- **元数据**: Survey / Collection Note | 整理时间 ≈ 2020 | **作者**: 社区整理
- **概述**: 实例分割（Instance Segmentation）方法综述，涵盖从 DeepMask、Mask R-CNN 到 BlendMask、FCIS、YOLACT 等全卷积方法的技术脉络。
- **关键要点**: 1. 实例分割的经典范式：两阶段（检测 → 分割）vs 单阶段（全卷积） 2. Mask R-CNN 是两阶段方法的基石，FPN 增强多尺度特征 3. BlendMask、YOLACT 等全卷积方法追求速度与精度的平衡
- **方法/发现**: 强监督方法（BoxeR、E2EC）和全景分割（Panoptic FPN）等延伸方向。
- **局限/意义**: 实例分割正从两阶段主导向单阶段高效方案演进，Transformer（如 Mask2Former）正在统一语义/实例/全景分割。

## 引用
- **阅读笔记**: [../../raw/cnn/Instance_Segmentation.md](../../raw/cnn/Instance_Segmentation.md)
