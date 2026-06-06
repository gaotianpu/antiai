---
id: liu_2017_rfb
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["RFB Net", "Receptive Field Block", "1711.07767"]
related_nodes: ["redmon_2016_yolov2", "liu_2018_panet"]
arxiv_id: 1711.07767
authors: Songtao Liu et al.
authors_institution: Beihang University
last_verified: 2026-06-06
---

# Receptive Field Block Net for Accurate and Fast Object Detection

- **元数据**: ECCV 2018 | arXiv | 2017 | **作者**: Songtao Liu, Di Huang, Yunhong Wang | **机构**: Beihang University
- **概述**: 受人类视觉感受野结构启发，提出感受野块（RFB）模块，通过考虑感受野大小与离心率的关系增强轻量级特征的判别能力，构建 RFB Net 检测器。
- **关键要点**: 1. RFB 模块模拟人类视觉中感受野大小随离心率增加的特性 2. 在 SSD 顶部挂载 RFB 模块，保持实时速度的同时达到深层骨干网络的精度 3. 主要设计元素：多分支空洞卷积 + 不同感受野尺度的组合
- **方法/发现**: 在 PASCAL VOC 和 MS COCO 上达到接近 ResNet-101 骨干检测器的精度，同时保持实时推理速度。
- **局限/意义**: RFB 是手工设计感受野增强的代表工作，后续被可变形卷积等自适应感受野方法超越。

## 引用
- **原始论文**: [arXiv:1711.07767](https://arxiv.org/abs/1711.07767) | [阅读笔记](../../raw/cnn/rfb.md)
