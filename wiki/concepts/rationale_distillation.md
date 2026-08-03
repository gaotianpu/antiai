---
id: rationale_distillation
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [理由蒸馏, 推理蒸馏]
related_nodes: [hsieh_2023_distilling, knowledge_distillation]
last_verified: 2026-08-03
---

# Rationale Distillation

## 定义
将 LLM 生成的推理过程（rationale）作为多任务训练信号蒸馏到小模型的范式：770M 的 T5 蒸馏 540B PaLM 的 rationale 后超越教师。

## 关键要点
- **信号形态**：rationale（推理链）比最终答案携带更多学习信号
- **多任务**：同一 rationale 可同时监督多个相关任务
- **成果**：小模型从推理过程而非仅答案中学习，性能反超

## 来源
- [[hsieh_2023_distilling]] — 理由蒸馏
