---
id: gevorgyan_2022_siou
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["SIoU", "SIoU Loss", "角度感知交并比损失", "2205.12740"]
related_nodes: ['iou_loss', 'iou_loss_survey']
arxiv_id: 2205.12740
authors: Zhora Gevorgyan
last_verified: 2026-09-30
---

# SIoU Loss: More Powerful Learning for Bounding Box Regression

- **元数据**: arXiv 2022 | **作者**: Zhora Gevorgyan | 相关: [[iou_loss]]
- **概述**: 提出 SIoU，在距离、形状、IoU 成本外加入角度成本，使预测框先朝最近的 x/y 轴对齐，再沿相关轴回归，解决预测框方向不确定、收敛慢的问题。
- **新颖概念**: [[iou_loss]]
- **关键要点**: 1. 角度成本 $\Lambda=1-2\sin^2(\arcsin x-\frac{\pi}{4})$，其中 $x=c_h/\sigma$ 2. 距离成本 $\Delta=\sum_{t=x,y}(1-e^{-\gamma\rho_t})$，$\gamma=2-\Lambda$ 3. 形状成本 $\Omega$ 与 IoU 成本共同构成 $L=1-IoU+\frac{\Delta+\Omega}{2}$ 4. 论文报告 COCO 上 +2.4% mAP@0.5:0.95、+3.6% mAP@0.5
- **方法/发现**: 将中心点连线的方向纳入惩罚，先减小角度自由度，再回归距离；在 COCO 训练中比 CIoU 收敛更快、精度更高。
- **局限/意义**: 角度和形状项引入额外超参数（形状参数 $\theta$ 常取 2–6），计算更复杂；后续 WIoU 在梯度分配机制上继续改进。

## 引用
- **原始论文**: [arXiv:2205.12740](https://arxiv.org/abs/2205.12740)
- **相关页面**: [[iou_loss]], [[iou_loss_survey]]
