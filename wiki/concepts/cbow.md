---
id: cbow
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [连续词袋, Continuous Bag-of-Words]
related_nodes: [word2vec_2013, skip_gram, word_embedding]
last_verified: 2026-08-03
---

# CBOW

## 定义
连续词袋（Continuous Bag-of-Words）模型：用上下文词的平均向量预测中心词，是 Word2Vec 的两种训练目标之一。

## 关键要点
- **方向**：上下文 → 中心词（与 Skip-gram 相反）
- **特点**：训练速度快，对高频词效果好
- **局限**：对罕见词的表示不如 Skip-gram

## 来源
- [[word2vec_2013]] — 提出 CBOW 与 Skip-gram 架构
