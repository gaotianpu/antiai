---
id: tong_2023_wiou
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["WIoU", "Wise-IoU", "动态聚焦交并比损失", "2301.10051"]
related_nodes: ['iou_loss', 'iou_loss_survey']
arxiv_id: 2301.10051
authors: Zanjia Tong et al.
last_verified: 2026-09-30
---

# Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism

- **元数据**: arXiv 2023 | **作者**: Zanjia Tong, Yuhang Chen, Zewei Xu, Rong Yu | 相关: [[iou_loss]]
- **概述**: 提出 WIoU 系列，用动态非单调聚焦机制按 anchor 质量分配梯度增益：减少高质量框的竞争，同时削弱低质量样本的有害梯度，使模型聚焦普通质量框。
- **新颖概念**: [[iou_loss]]
- **关键要点**: 1. WIoU v1：$L_{WIoUv1}=R_{WIoU}L_{IoU}$，$R_{WIoU}=\exp(\frac{(x-x^{gt})^2+(y-y^{gt})^2}{(W_g^2+H_g^2)^*})$ 2. v2 加入单调聚焦系数，并用滑动平均归一化梯度增益 3. v3 以离群度 $\beta=\frac{L^*_{IoU}}{\overline{L_{IoU}}}$ 构造非单调系数 $r=\frac{\beta}{\delta\alpha^{\beta-\delta}}$ 4. 应用于 YOLOv7 时 AP75 从 53.03% 提升到 54.50%
- **方法/发现**: 将 BBR 损失写为 $L=L_{IoU}+R$ 的范式，用 attention-based 距离项和动态聚焦系数调节每个 anchor 的梯度；在模拟实验与 COCO 上验证优于 SIoU/EIoU/Focal-EIoU。
- **局限/意义**: 需要维护 $\overline{L_{IoU}}$ 滑动平均并调节 $\alpha,\delta,\gamma$ 等超参数；动态机制对训练前期策略敏感。

## 引用
- **原始论文**: [arXiv:2301.10051](https://arxiv.org/abs/2301.10051) | [代码](https://github.com/Instinct323/wiou) | [阅读笔记](../../raw/2301.10051.md)
- **相关页面**: [[iou_loss]], [[iou_loss_survey]]
