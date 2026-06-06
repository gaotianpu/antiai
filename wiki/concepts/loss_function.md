---
id: loss_function
type: concept
tags: ["machine-learning", "theoretical"]
aliases: ["损失函数", "loss", "目标函数", "代价函数"]
related_nodes: ["optimizer", "focal_loss_2017", "object_detection", "probability_distributions", "entropy", "kl_divergence"]
last_verified: 2026-06-06
---

# Loss Function（损失函数）

## 定义
损失函数衡量模型预测与真实标签之间的差异，是优化目标的核心。不同任务需要不同的损失函数设计。

## 分类

### 回归损失
| 损失 | 公式 | 特点 | 出处 |
|:---|:---|:---|:---|
| **MSE (L2)** | (y - ŷ)² | 对异常值敏感，梯度线性 | 经典 |
| **MAE (L1)** | \|y - ŷ\| | 对异常值鲁棒，零点不可导 | 经典 |
| **Huber** | 结合 L1+L2 | 平滑过渡，兼顾两者优势 | Huber, 1964 |

### 分类损失
| 损失 | 公式 | 特点 | 出处 |
|:---|:---|:---|:---|
| **交叉熵 (CE)** | -∑y·log(ŷ) | 分类任务标准，配合 Softmax | 信息论 |
| **二元交叉熵 (BCE)** | -[y·log(ŷ)+(1-y)·log(1-ŷ)] | 二分类标准 | — |
| **Focal Loss** | -(1-ŷ)ᵞ·log(ŷ) | 缓解类别不平衡，难例聚焦 | [[focal_loss_2017]] |

### 度量学习损失
| 损失 | 核心思想 | 出处 |
|:---|:---|:---|
| **Contrastive Loss** | 相似对拉近，不相似对推远 | LeCun et al., 2005 |
| **Triplet Loss** | 锚点+正例+负例三元组约束 | FaceNet, 2015 |
| **InfoNCE** | 对比学习中的噪声对比估计 | CPC, 2018 |

### 检测/分割损失
| 损失 | 核心思想 | 出处 |
|:---|:---|:---|
| **IoU / GIoU / CIoU** | 边界框交并比及改进变体 | YOLO 系列 |
| **Dice Loss** | 分割中重叠区域度量 | Milletari et al., 2016 |
| **Smooth L1** | 检测框回归的 Huber 变体 | Faster R-CNN |

## 关键发现
- Cross-Entropy 是分类任务默认选择，配合 Softmax 输出
- Focal Loss 解决一阶段检测器的极端正负样本不平衡（[[focal_loss_2017]]）
- IoU 系列损失比 L1/L2 更直接优化检测目标
- InfoNCE 是 CLIP/SimCLR 等对比学习方法的核心

## 相关概念
- [[optimizer]] — 优化器：最小化损失函数的算法
- [[object_detection]] — 目标检测中的多任务损失
