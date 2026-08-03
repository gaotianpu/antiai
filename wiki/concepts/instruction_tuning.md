---
id: instruction_tuning
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [指令微调, 指令调优]
related_nodes: [ouyang_2022_instructgpt, wang_2022_selfinstruct, reinforcement_learning_from_human_feedback]
last_verified: 2026-08-03
---

# Instruction Tuning

## 定义
用"指令-回答"形式的数据集对预训练模型进行监督微调，使模型学会遵循人类指令，显著提升可用性与泛化能力。

## 关键要点
- **数据形态**：多样化任务 × 自然语言指令描述
- **与 RLHF 的关系**：指令微调（SFT）是 RLHF 的前置步骤
- **数据来源**：人工标注（InstructGPT）或模型自生成（Self-Instruct）

## 来源
- [[ouyang_2022_instructgpt]] — InstructGPT 指令微调
- [[wang_2022_selfinstruct]] — 自我指令数据生成
