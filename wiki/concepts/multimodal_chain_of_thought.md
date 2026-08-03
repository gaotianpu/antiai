---
id: multimodal_chain_of_thought
type: concept
tags: [NLP, computer-vision, machine-learning, empirical-study]
aliases: [多模态思维链, Multimodal CoT]
related_nodes: [zhang_2023_multimodal_cot, chain_of_thought]
last_verified: 2026-08-03
---

# Multimodal Chain-of-Thought

## 定义
将 CoT 推理扩展到多模态场景：两阶段框架先基于图文生成推理原理，再据原理得出答案，融合文本与图像信息提升推理质量。

## 关键要点
- **两阶段**：原理生成（rationale）→ 答案推理，避免一步到位
- **多模态融合**：文本 token + 图像特征共同输入
- **效果**：ScienceQA 等基准上显著超越单模态 CoT

## 来源
- [[zhang_2023_multimodal_cot]] — Multimodal CoT
