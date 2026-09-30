---
id: rezatofighi_2019_giou
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["GIoU", "Generalized IoU", "广义交并比", "1902.09630"]
related_nodes: ['iou_loss', 'iou_loss_survey']
arxiv_id: 1902.09630
authors: Hamid Rezatofighi et al.
last_verified: 2026-09-30
---

# Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression

- **元数据**: CVPR 2019 | **作者**: Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, Silvio Savarese | 相关: [[iou_loss]]
- **概述**: 提出 GIoU，通过最小闭包区域 $C$ 为不相交框提供梯度；GIoU 同时是度量与损失，范围 $[-1,1]$，在 Faster R-CNN、Mask R-CNN、YOLOv3 上一致提升。
- **新颖概念**: [[iou_loss]]
- **关键要点**: 1. $GIoU = IoU - \frac{|C\setminus(A\cup B)|}{|C|}$ 2. $C$ 是同时包含预测框与 GT 的最小闭包框 3. 不相交时仍有非零值和梯度，缓解 IoU 平台问题 4. 作为度量可比较任意两形状，作为损失可直接嵌入主流检测器
- **方法/发现**: 推导轴对齐矩形的 GIoU 解析形式，并替换检测器中的回归损失；在 PASCAL VOC 与 MS COCO 上验证了一致增益。
- **局限/意义**: 当一个框包含另一个框时，GIoU 退化为 IoU，无法区分相对位置；且依赖闭包框计算，后续 DIoU/CIoU 继续改进。

## 引用
- **原始论文**: [CVPR 2019 / arXiv:1902.09630](https://arxiv.org/abs/1902.09630)
- **相关页面**: [[iou_loss]], [[iou_loss_survey]]
