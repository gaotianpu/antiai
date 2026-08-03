---
id: gpt_2
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [GPT-2, 15亿参数语言模型]
related_nodes: [radford_2019_gpt2, gpt, zero_shot_learning]
last_verified: 2026-08-03
---

# GPT-2

## 定义
GPT 系列第二代：将参数量扩展至 15 亿，证明语言模型在无监督预训练后可零样本执行翻译、问答、摘要等多种下游任务。

## 关键要点
- **零样本能力**：无需任何任务数据微调即可完成任务，颠覆"必须微调"的认知
- **规模效应**：参数量越大，零样本性能越强，预示 Scaling Law
- **方法论**：任务以自然语言形式描述，模型按提示条件生成

## 来源
- [[radford_2019_gpt2]] — GPT-2 原始论文
