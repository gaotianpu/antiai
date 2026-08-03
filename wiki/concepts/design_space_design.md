---
id: design_space_design
type: concept
tags: [machine-learning, theoretical]
aliases: [设计空间设计, 网络设计空间]
related_nodes: [radosavovic_2020_regnet, regnet]
last_verified: 2026-08-03
---

# Design Space Design

## 定义
RegNet 提出的方法论：不是设计单个网络，而是构建"网络设计空间"（参数化分布），通过空间采样与统计评估发现好架构的共性规律。

## 关键要点
- **从实例到空间**：评估空间的整体质量（误差分布）而非单点
- **渐进缩小**：从大空间通过分析逐步收缩到高质量低复杂度空间
- **发现**：最优深度/宽度/瓶颈比呈简单函数规律，可手工复现

## 来源
- [[radosavovic_2020_regnet]] — RegNet 设计空间方法论
