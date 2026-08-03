---
id: decoupled_head
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [解耦头, 分类回归解耦]
related_nodes: [ge_2021_yolox, single_stage_detector]
last_verified: 2026-08-03
---

# Decoupled Head

## 定义
将检测头的分类与回归分支拆分为独立子网络的设计：分类关注"是什么"（类间可分），回归关注"在哪"（空间定位），任务冲突时解耦更优。

## 关键要点
- **任务冲突**：分类与回归优化目标差异大，共享头相互干扰
- **YOLOX 发现**：解耦头显著提升收敛速度与精度
- **代价**：参数量与延迟略增

## 来源
- [[ge_2021_yolox]] — YOLOX 解耦头
