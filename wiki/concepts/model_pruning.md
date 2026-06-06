---
id: model_pruning
type: concept
tags: [machine-learning, empirical-study, parameter-optimization]
aliases: [剪枝, pruning]
related_nodes: [han_2015_deepcompression, cheng_2017_modelcompression]
last_verified: 2026-06-06
---

# Model Pruning

## 定义
Model Pruning（模型剪枝）通过移除神经网络中不重要的权重、神经元或通道来减小模型尺寸和计算量，同时尽量保持精度。

## 剪枝类型
| 类型 | 粒度 | 效果 |
|:---|:---|:---|
| 非结构化剪枝 | 单个权重 | 高压缩率但需专用硬件 |
| 结构化剪枝 | 通道/层 | 直接加速，通用硬件友好 |
| 迭代剪枝 | 剪枝→微调循环 | 逐步恢复精度 |

## 关键原则
- **Lottery Ticket Hypothesis**（Frankle & Carbin 2019）— 存在与原始网络精度相当的稀疏子网络
- **重要性准则**：基于权重幅度、梯度或 Hessian 的剪枝标准

## 来源
- [[han_2015_deepcompression]] — 深度压缩：剪枝+量化+哈夫曼编码
- [[cheng_2017_modelcompression]] — 模型压缩综述
