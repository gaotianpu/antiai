---
id: multimodal_language_model
type: concept
tags: [NLP, computer-vision, machine-learning, empirical-study]
aliases: [多模态语言模型, MLLM]
related_nodes: [kosmos_1, multimodal_fusion, instruction_tuning]
last_verified: 2026-08-03
---

# Multimodal Language Model

## 定义
以语言模型为核心、融合图像/音频等多模态输入的模型：将非文本模态 token 化后接入 LLM，统一理解与生成。

## 关键要点
- **架构范式**：模态编码器 + 投影层 + LLM 骨干
- **训练阶段**：模态对齐预训练 → 指令微调
- **代表**：Kosmos-1、GPT-4V、LLaVA 等

## 来源
- [[kosmos_1]] — Kosmos-1 多模态语言模型
