---
id: self_instruct
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [自我指令, Self-Instruct]
related_nodes: [wang_2022_selfinstruct, instruction_tuning]
last_verified: 2026-08-03
---

# Self-Instruct

## 定义
让 LLM 自我生成指令数据并指导自身微调的流程：种子指令 → 模型生成任务 → 过滤 → 微调，大幅降低指令数据的标注成本。

## 关键要点
- **四步闭环**：种子集、生成、过滤（质量/多样性）、微调迭代
- **效果**：仅用少量种子即可获得与人工指令数据相当的模型
- **意义**：开启"模型自举数据"的数据飞轮范式，Alpaca 等后续工作直接沿用

## 来源
- [[wang_2022_selfinstruct]] — Self-Instruct 原始论文
