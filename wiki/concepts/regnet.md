---
id: regnet
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [RegNet, 正则化网络族]
related_nodes: [radosavovic_2020_regnet, design_space_design]
last_verified: 2026-08-03
---

# RegNet

## 定义
从设计空间中系统发现的网络家族：深度/宽度/瓶颈比按简单线性函数缩放，以较低 FLOPs 达到超越 EfficientNet 的精度-效率权衡。

## 关键要点
- **规律性**：最优网络参数遵循可解析的缩放规律（无需搜索即可生成）
- **效率**：同 FLOPs 下精度优于当时 SOTA
- **意义**：证明"设计空间统计"可替代"单点暴力搜索"

## 来源
- [[radosavovic_2020_regnet]] — RegNet 家族
