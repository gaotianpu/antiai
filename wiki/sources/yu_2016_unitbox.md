---
id: yu_2016_unitbox
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["UnitBox", "IoU Loss", "交并比损失", "1608.01471"]
related_nodes: ['iou_loss', 'iou_loss_survey']
arxiv_id: 1608.01471
authors: Jiahui Yu et al.
last_verified: 2026-09-30
---

# UnitBox: An Advanced Object Detection Network

- **元数据**: ACM MM 2016 | **作者**: Jiahui Yu, Yuning Jiang, Zhangyang Wang, Zhimin Cao, Thomas Huang | 相关: [[iou_loss]]
- **概述**: 首次把 IoU 作为边界框回归损失，将预测框四边作为整体优化，替代把坐标当独立变量的 L2 回归；在 FDDB 人脸检测上取得当时最优结果。
- **新颖概念**: [[iou_loss]]
- **关键要点**: 1. 动机：L2 独立回归四个坐标，与定位质量不一致 2. 原文 IoU 损失为 $L_{IoU}=-\ln(\frac{Intersection}{Union})$ 3. 后续通用形式为 $L_{IoU}=1-IoU$ 4. UnitBox 用全卷积网络做像素级边框预测，端到端优化
- **方法/发现**: 将边界框视为一个整体单元，直接最大化预测框与 GT 的交并比；对物体形状和尺度变化更鲁棒，收敛更快。
- **局限/意义**: 两框不重叠时 $IoU=0$、梯度消失；这成为 GIoU、DIoU 等后续改进的出发点。IoU 作为评测指标则可追溯到 [[everingham_2010_pascal_voc]] 所代表的 PASCAL VOC 协议。

## 引用
- **原始论文**: [ACM MM 2016, DOI:10.1145/2964284.2967274](https://doi.org/10.1145/2964284.2967274) | [arXiv:1608.01471](https://arxiv.org/abs/1608.01471) | [阅读笔记](../../raw/1608.01471.md)
- **相关页面**: [[iou_loss]], [[iou_loss_survey]], [[everingham_2010_pascal_voc]]
