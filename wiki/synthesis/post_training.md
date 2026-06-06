---
id: post_training
type: synthesis
tags: ["practical-guide", "machine-learning", "NLP"]
aliases: ["后训练", "post-training", "post training alignment"]
related_nodes: ["openai_2023_gpt4", "reinforcement_learning_from_human_feedback"]
---

# Post-Training（后训练）

## 概述

Post-training 指预训练完成之后、模型部署之前的所有优化环节。其核心目标是在**不显著改变基础模型能力**的前提下，提升模型在特定维度上的表现，包括对齐人类偏好、安全性、推理效率等。

## 主要技术路线

| 技术 | 目标 | 典型应用 |
|:---|:---|:---|
| **RLHF**（[[reinforcement_learning_from_human_feedback\|人类反馈强化学习]]） | 对齐人类偏好，提升有用性与真实性 | [[openai_2023_gpt4]] |
| **Constitutional AI** | 无需人工标注实现自我修正 | Dromedary |
| **PTQ / QAT（量化）** | 降低推理延迟和显存占用 | YOLOv6、MCUNet |

## 关键发现

- **数据量极小**：post-training 数据集远小于预训练集，却能大幅改善模型行为（[[openai_2023_gpt4]]）
- **基础能力几乎不变**：GPT-4 在 RLHF 后基准测试得分从 73.7% 微升至 74.0%（[[openai_2023_gpt4]]）
- **校准度可能下降**：预训练模型校准良好，但 post-training 后校准度降低（[[openai_2023_gpt4]]）

## 相关概念

- [[reinforcement_learning_from_human_feedback|RLHF / 人类反馈强化学习]]
- AI 对齐 (alignment)
- 微调 (fine-tuning)
- 量化 (quantization)
