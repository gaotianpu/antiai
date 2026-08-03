---
id: skip_gram
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [跳元模型, Skip-gram]
related_nodes: [word2vec_2013, cbow, word_embedding]
last_verified: 2026-08-03
---

# Skip-gram

## 定义
用中心词预测上下文窗口内的词，是 Word2Vec 的两种训练目标之一，对低频词和短语语义的建模优于 CBOW。

## 关键要点
- **方向**：中心词 → 上下文（与 CBOW 相反）
- **特点**：对罕见词更鲁棒，但训练较慢
- **负采样**：通过采样负例避免遍历全词表，大幅加速训练

## 来源
- [[word2vec_2013]] — 提出 Skip-gram 与负采样
