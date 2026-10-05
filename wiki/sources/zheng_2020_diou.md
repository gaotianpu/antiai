---
id: zheng_2020_diou
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["DIoU", "CIoU", "Distance-IoU", "Complete IoU", "距离交并比", "1911.08287"]
related_nodes: ['iou_loss', 'iou_loss_survey']
arxiv_id: 1911.08287
authors: Zhaohui Zheng et al.
last_verified: 2026-09-30
---

# Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression

- **元数据**: AAAI 2020 | **作者**: Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, Dongwei Ren | 相关: [[iou_loss]]
- **概述**: 提出 DIoU 与 CIoU：DIoU 加入中心点归一化距离惩罚，收敛快于 IoU/GIoU；CIoU 在此基础上加入长宽比一致性项，同时考虑重叠、中心与形状。
- **新颖概念**: [[iou_loss]]
- **关键要点**: 1. $L_{DIoU}=1-IoU+\frac{\rho^2(b,b^{gt})}{c^2}$，$c$ 为闭包框对角线长 2. 总结 BBR 的三个几何因素：重叠面积、中心距离、长宽比 3. CIoU 加 $v=\frac{4}{\pi^2}(\arctan\frac{w^{gt}}{h^{gt}}-\arctan\frac{w}{h})^2$ 与权重 $\alpha$ 4. 在 YOLOv4/v5 等工业检测器中广泛使用
- **方法/发现**: 通过中心距离惩罚直接拉近预测框与 GT 的中心，解决 IoU/GIoU 收敛慢的问题；CIoU 提升形状一致性，在 PASCAL VOC 与 COCO 上稳定涨点。
- **局限/意义**: CIoU 的长宽比项对宽高梯度强耦合，且同一比例不同尺寸的框惩罚可能不够敏感；EIoU 将宽、高拆开分别惩罚来改进。

## 引用
- **原始论文**: [AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6916) | [arXiv:1911.08287](https://arxiv.org/abs/1911.08287) | [阅读笔记](../../raw/1911.08287.md)
- **相关页面**: [[iou_loss]], [[iou_loss_survey]]
