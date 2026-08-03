---
id: cheap_operation
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [廉价操作, 低成本变换]
related_nodes: [han_2019_ghostnet, ghost_module]
last_verified: 2026-08-03
---

# Cheap Operation

## 定义
GhostNet 的设计原则：用计算成本极低的线性变换（如 depthwise 卷积）生成冗余特征图，替代昂贵的标准卷积。

## 关键要点
- **直觉**：特征图相似性高，冗余部分可由少量核心特征派生
- **约束**：线性变换保证与核心特征的相关性
- **意义**：将"特征冗余"从副作用转为设计资源

## 来源
- [[han_2019_ghostnet]] — Ghost 模块的设计基础
