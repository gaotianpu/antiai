---
id: ren_2015_fasterrcnn
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["Faster R-CNN", "RPN", "Region Proposal Network", "1506.01497"]
related_nodes: ["girshick_2013_rcnn"]
arxiv_id: 1506.01497
authors: Shaoqing Ren et al.
authors_institution: Microsoft Research
last_verified: 2026-06-06
---

# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

- **元数据**: arXiv | 2015 | **作者**: Shaoqing Ren et al. | **机构**: Microsoft Research | 相关: [[girshick_2013_rcnn]]
- **概述**: 提出区域候选网络（RPN），将候选框生成融入检测网络实现端到端训练，近乎实时检测。
- **关键要点**: 1. RPN 与 Fast R-CNN 共享卷积特征 2. 锚框机制 3. 端到端训练
- **方法/发现**: 用 RPN 替代 Selective Search，将候选框生成成本降低到近乎零
- **局限/意义**: 两阶段检测器标准范式，RPN 锚框思想被 YOLO/SSD 借鉴

## 引用
- **原始论文**: [arXiv:1506.01497](https://arxiv.org/abs/1506.01497) | [阅读笔记](../../raw/cnn/Faster_R-CNN.md)
- **相关概念**: [[girshick_2013_rcnn]]
