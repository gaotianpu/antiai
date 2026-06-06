---
id: semantic_segmentation_survey
type: source
tags: ["computer-vision", "machine-learning", "survey"]
aliases: ["语义分割综述", "Semantic Segmentation"]
related_nodes: ["long_2014_fcn", "sppnet_2014"]
arxiv_id: null
authors: null
authors_institution: null
last_verified: 2026-06-06
---

# 语义分割 Semantic Segmentation

- **元数据**: Survey / Collection Note | 整理时间 ≈ 2020 | **作者**: 社区整理
- **概述**: 语义分割（Semantic Segmentation）方法综述，涵盖从 FCN、U-Net、SegNet 到 DeepLab、PSPNet 的关键技术路径。
- **新颖概念**: —
- **关键要点**: 1. 三大思路：滑窗、候选区域、编码器-解码器 2. FCN 奠基端到端像素预测，U-Net 引入对称编码器-解码器结构 3. DeepLab 系列以空洞卷积和 ASPP 实现多尺度上下文聚合
- **方法/发现**: 主要数据集：PASCAL VOC、Cityscapes、MS COCO。评估指标：Pixel Acc、mAcc、mIoU、FWIoU。
- **局限/意义**: 语义分割正向 Transformer（SETR、SegFormer）和掩码范式（MaskFormer）演进，CNN 时代的经典方法仍为高效基座。

## 引用
- **阅读笔记**: [../../raw/cnn/Semantic_Segmentation.md](../../raw/cnn/Semantic_Segmentation.md)
