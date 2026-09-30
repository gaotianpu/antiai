---
id: everingham_2010_pascal_voc
type: source
tags: ["computer-vision", "machine-learning", "survey"]
aliases: ["PASCAL VOC", "VOC 2010", "PASCAL VOC 目标检测挑战赛", "The PASCAL Visual Object Classes Challenge"]
related_nodes: ['iou_loss', 'iou_loss_survey', 'yu_2016_unitbox']
arxiv_id: null
authors: Mark Everingham et al.
last_verified: 2026-09-30
---

# The PASCAL Visual Object Classes (VOC) Challenge

- **元数据**: IJCV 2010 | **作者**: Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, Andrew Zisserman | 相关: [[iou_loss]]
- **概述**: PASCAL VOC 系列挑战赛总结，定义 20 类目标检测的标注规范、数据划分与评估协议，并把 IoU 阈值匹配 + mAP 固化为检测评测标准。
- **新颖概念**: —
- **关键要点**: 1. 将预测框与 GT 的 IoU ≥ 阈值才视为命中 2. 统一 AP/mAP 计算流程，使不同检测器可公平比较 3. 覆盖分类、检测、分割等任务与难度分析 4. 后续 MS COCO 等基准延续并扩展该评估范式
- **方法/发现**: 系统描述 PASCAL VOC 系列挑战赛的数据集构建、标注流程、评估指标（precision、recall、AP、mAP）与结果分析。
- **局限/意义**: IoU 作为评估指标不可导，且两框不重叠时无梯度；评测标准与回归损失之间的 gap 直接催生 GIoU、DIoU、CIoU 等后续工作。

## 引用
- **原始论文**: [IJCV 2010, DOI:10.1007/s11263-009-0275-4](https://doi.org/10.1007/s11263-009-0275-4)
- **相关页面**: [[iou_loss]], [[iou_loss_survey]]
