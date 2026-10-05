---
id: zhang_2022_eiou
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["EIoU", "Focal-EIoU", "Efficient IoU", "高效交并比损失", "2101.08158"]
related_nodes: ['iou_loss', 'iou_loss_survey']
arxiv_id: 2101.08158
authors: Yi-Fan Zhang et al.
last_verified: 2026-09-30
---

# Focal and Efficient IOU Loss for Accurate Bounding Box Regression

- **元数据**: Neurocomputing 2022 | **作者**: Yi-Fan Zhang, Weiqiang Ren, Zhang Zhang, Zhen Jia, Liang Wang, Tieniu Tan | 相关: [[iou_loss]]
- **概述**: 提出 EIoU，将 CIoU 的长宽比耦合惩罚拆成宽、高分别惩罚；进一步提出 Focal-EIoU，用 $IoU^\gamma$ 抑制大量低重叠 anchor 对回归的主导。
- **新颖概念**: [[iou_loss]]
- **关键要点**: 1. $L_{EIoU}=1-IoU+\frac{\rho^2(b,b^{gt})}{c_w^2+c_h^2}+\frac{\rho^2(w,w^{gt})}{c_w^2}+\frac{\rho^2(h,h^{gt})}{c_h^2}$ 2. 直接最小化宽、高差异，比 CIoU 更稳定、更细粒度 3. 指出 BBR 中存在大量小重叠 anchor 主导优化的问题 4. $L_{Focal-EIoU}=IoU^\gamma L_{EIoU}$，抑制低质量 anchor，聚焦高质量/普通质量框
- **方法/发现**: 在合成数据与真实检测器上验证 EIoU 收敛更快、定位更准；Focal-EIoU 将回归损失与 focal 式重加权结合，γ 控制离群 anchor 的抑制程度。
- **局限/意义**: Focal-EIoU 与分类 Focal Loss 机制不同：这里用 $IoU^\gamma$ 降权低重叠框，而非直接给难例加权；γ 需调参，不同检测器上增益不稳定。

## 引用
- **原始论文**: [Neurocomputing 2022, DOI:10.1016/j.neucom.2022.07.042](https://doi.org/10.1016/j.neucom.2022.07.042) | [arXiv:2101.08158](https://arxiv.org/abs/2101.08158) | [阅读笔记](../../raw/2101.08158.md)
- **相关页面**: [[iou_loss]], [[iou_loss_survey]]
