---
id: zero_shot_chain_of_thought
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [零样本思维链, Zero-shot CoT]
related_nodes: [zero_shot_cot_2022, chain_of_thought, few_shot_learning]
last_verified: 2026-08-03
---

# Zero-Shot Chain-of-Thought

## 定义
零样本 CoT：无需人工示例，仅追加一句 "Let's think step by step" 即可激发 LLM 逐步推理，在算术/常识/符号推理任务上大幅提升。

## 关键要点
- **触发短语**：简单的推理引导句激活模型中间推理
- **零标注成本**：对比 few-shot CoT 无需设计示例
- **局限**：推理步骤可能出错，置信度校准需改进

## 来源
- [[zero_shot_cot_2022]] — Zero-shot CoT
