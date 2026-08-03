---
id: self_align
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [自我对齐, SELF-ALIGN]
related_nodes: [sun_2023_dromedary, principle_driven_alignment, reinforcement_learning_from_human_feedback]
last_verified: 2026-08-03
---

# SELF-ALIGN

## 定义
无需 RLHF 的模型自对齐方法：以 16 条人工原则为基础，通过上下文学习让 LLM 自我生成对齐示范并对齐自身，仅需 <300 行人工标注。

## 关键要点
- **原则驱动**：16 条原则涵盖有用性/诚实性/无害性
- **三步流程**：原则+种子示范 → 上下文学习生成 → 自我反思优化
- **成果**：Dromedary 达到与 InstructGPT 相当的对齐水平

## 来源
- [[sun_2023_dromedary]] — SELF-ALIGN
