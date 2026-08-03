---
id: knowledge_enhanced_pretraining
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [知识增强预训练, KEP]
related_nodes: [ernie_2019, bert, masked_language_modeling]
last_verified: 2026-08-03
---

# Knowledge-Enhanced Pretraining

## 定义
将外部知识（知识图谱实体、关系）注入预训练过程的范式：在 MLM 等目标基础上加入知识感知的掩码/预测任务，提升语言理解的常识与事实能力。

## 关键要点
- **实体级掩码**：掩码整个实体而非单个词，迫使模型调用知识
- **知识对齐**：把文本 token 与知识图谱实体对齐训练
- **代表工作**：ERNIE 系列、KnowBERT 等

## 来源
- [[ernie_2019]] — 知识增强预训练 ERNIE
