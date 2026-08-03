---
id: implicit_knowledge
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [隐式知识, 隐式编码]
related_nodes: [wang_2021_yolor, knowledge_distillation]
last_verified: 2026-08-03
---

# Implicit Knowledge

## 定义
YOLOR 提出的知识形态：模型在推理时以隐式（不可直接观察的）表征携带的任务知识，与显式知识（网络参数/特征）共同构成统一表征。

## 关键要点
- **显隐互补**：显式知识可观察可解释，隐式知识难以直接解释但提升性能
- **实现**：隐式模块（可学习的知识表示）嵌入网络各层
- **统一表征**：多任务共享一套显隐结合的编码

## 来源
- [[wang_2021_yolor]] — YOLOR
