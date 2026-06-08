---
id: chain_of_thought
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [思维链, CoT, chain of thought prompting]
related_nodes: [in_context_learning, deepseek_2025_r1]
---

# Chain-of-Thought

## 概述

Chain-of-Thought（思维链，CoT）是 prompting 的一种中间推理方法，通过在提示中展示逐步推理步骤（intermediate reasoning steps），引导大语言模型生成更准确、更可解释的推理链。

## 详细阐述

### 定义与内涵

CoT 的核心假设是：复杂推理任务（数学、逻辑、符号）难以直接映射到答案，但若模型先产生中间推理步骤，再从中推导最终答案，准确率显著提升。CoT 属于 [[in_context_learning]] 的延伸——它在演示示例中不仅包含「输入-输出」，还包含「输入-推理过程-输出」。

### 主要方法

| 方法 | 核心设计 | 出处 |
|:---|:---|:---|
| CoT Prompting | 提示中给出逐步推理的 few-shot 示例 | [[wei_2022_cot]] |
| Zero-shot CoT | 用 "Let's think step by step" 触发推理 | [[zero_shot_cot_2022]] |
| Auto-CoT | 自动生成推理链，减少人工演示依赖 | [[auto_cot_2022]] |
| Multimodal CoT | 扩展到多模态输入（文本+图像） | [[zhang_2023_multimodal_cot]] |

### 关键维度

- **复杂性问题**：CoT 在算术、常识推理、符号推理任务上提升最大
- **模型规模效应**：CoT 的有效性与模型规模正相关（emergent ability），小模型从 CoT 中获益有限
- **推理链格式**：自由文本 vs 结构化格式（如 JSON 中间步骤）
- **集成方法**：Self-Consistency（采样多条推理链后投票）、Tree-of-Thought（分支探索）

### 局限性

- 推理链不能保证事实正确，可能产生连贯但错误的过程
- 增加 token 消耗和推理延迟
- 对 prompt 格式敏感

## 相关概念网络

- [[in_context_learning]] — CoT 是 ICL 的一种特殊形式
- [[attention_mechanism]] — 注意力机制支撑长程推理链处理

## 引用资料

1. [[wei_2022_cot]] — CoT Prompting 原始论文
2. [[zero_shot_cot_2022]] — Zero-shot CoT
3. [[auto_cot_2022]] — Automation CoT
4. [[zhang_2023_multimodal_cot]] — Multimodal CoT

## 更新记录

- **2026-06-06**: 初始创建
