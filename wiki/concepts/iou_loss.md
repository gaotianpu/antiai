---
id: iou_loss
type: concept
tags: ["computer-vision", "machine-learning", "theoretical"]
aliases: ["IoU Loss", "交并比损失", "IoU 损失", "Bounding Box Regression Loss"]
related_nodes: ['everingham_2010_pascal_voc', 'yu_2016_unitbox', 'rezatofighi_2019_giou', 'zheng_2020_diou', 'zhang_2022_eiou', 'gevorgyan_2022_siou', 'tong_2023_wiou', 'he_2021_alpha_iou', 'iou_loss_survey', 'object_detection', 'loss_function']
last_verified: 2026-09-30
---

# IoU Loss（交并比损失）

## 定义
IoU Loss 是以预测框与真实框的交并比（Intersection over Union）为核心构造的边界框回归损失族。它直接优化定位质量，而不是像 L2/L1 那样独立回归四个坐标。

## 关键变体

| 变体 | 核心机制 | 解决/改善的问题 | 来源 |
|:---|:---|:---|:---|
| **IoU** | $1-IoU$（UnitBox 原式为 $-\ln IoU$） | 基础重合度，整体回归 | [[yu_2016_unitbox]] |
| **GIoU** | $IoU-\frac{\lvert C\setminus(A\cup B)\rvert}{\lvert C\rvert}$ | 不相交时无梯度、范围 $[-1,1]$ | [[rezatofighi_2019_giou]] |
| **DIoU** | $1-IoU+\frac{\rho^2(b,b^{gt})}{c^2}$ | 中心距离惩罚，加快收敛 | [[zheng_2020_diou]] |
| **CIoU** | DIoU + 长宽比一致性项 $v$ | 同时考虑重叠、中心、形状 | [[zheng_2020_diou]] |
| **EIoU** | 宽、高分别惩罚 | 解耦长宽比，比 CIoU 更稳定 | [[zhang_2022_eiou]] |
| **SIoU** | 加入角度成本 $\Lambda$ | 先朝最近轴对齐，减少方向游荡 | [[gevorgyan_2022_siou]] |
| **WIoU** | 动态非单调聚焦机制 | 按 anchor 质量分配梯度 | [[tong_2023_wiou]] |
| **Alpha-IoU** | $(1-IoU^\alpha)/\alpha$ | 幂次推广，统一并调节多种损失 | [[he_2021_alpha_iou]] |
| **Focal-EIoU** | $IoU^\gamma L_{EIoU}$ | 抑制低质量 anchor 主导 | [[zhang_2022_eiou]] |

## 关键要点
- **度量与损失**：IoU 作为评测指标由 [[everingham_2010_pascal_voc]] 等检测基准推广；作为回归损失首见于 [[yu_2016_unitbox]]。
- **改进维度**：非重叠梯度（GIoU）、中心距离（DIoU）、长宽形状（CIoU/EIoU）、方向角度（SIoU）、样本/ anchor 质量加权（Focal-EIoU/WIoU）、损失族幂次统一（Alpha-IoU）。
- **实践常用**：CIoU 和 EIoU 是通用检测基线；SIoU/WIoU 常用于追求更高定位精度；Alpha-IoU 适合统一实验和鲁棒性分析。
- **命名陷阱**：Focal-EIoU 与分类 Focal Loss 机制不同；WIoU 的“聚焦”是动态非单调梯度分配，并非简单给难例加权。

## 来源
- [[everingham_2010_pascal_voc]] — IoU 作为检测评测协议的标准化
- [[yu_2016_unitbox]] — 首次提出 IoU loss
- [[rezatofighi_2019_giou]] — GIoU
- [[zheng_2020_diou]] — DIoU / CIoU
- [[zhang_2022_eiou]] — EIoU / Focal-EIoU
- [[gevorgyan_2022_siou]] — SIoU
- [[tong_2023_wiou]] — WIoU v1/v2/v3
- [[he_2021_alpha_iou]] — Alpha-IoU 损失族

## 更新记录
- **2026-09-30**: 初始创建，汇总 IoU 及其主要改进变体与对应 Source 页。
