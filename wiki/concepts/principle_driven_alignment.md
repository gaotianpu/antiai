---
id: principle_driven_alignment
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [原则驱动对齐, SELF-ALIGN]
related_nodes: [sun_2023_dromedary, self_align, reinforcement_learning_from_human_feedback]
last_verified: 2026-08-03
---

# Principle-Driven Alignment

## 定义
SELF-ALIGN 提出的对齐范式：用 16 条人工原则 + 上下文学习让模型自我对齐（SELF-ALIGN），仅需 <300 行人工标注即可获得与 RLHF 相当的对齐效果。

## 关键要点
- **原则而非偏好**：以原则集替代大规模人类偏好标注
- **三步流程**：原则定义 → 上下文学习生成对齐示范 → 模型自我反思优化
- **意义**：探索"无 RLHF"的低成本对齐路径

## 来源
- [[sun_2023_dromedary]] — SELF-ALIGN 方法
