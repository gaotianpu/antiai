---
id: word_embedding
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [词向量, 词嵌入, word vector]
related_nodes: [word2vec_2013, cbow, skip_gram, tokenization]
last_verified: 2026-08-03
---

# Word Embedding

## 定义
将离散词符映射为稠密低维实数向量，使语义相近的词在向量空间中距离相近，是 NLP 表示学习的基础。

## 关键要点
- **分布假设**：词的语义由其上下文决定（"You shall know a word by the company it keeps"）
- **静态嵌入**：Word2Vec/GloVe 为每个词学习单一向量，无法消歧
- **上下文嵌入**：BERT 等预训练模型按上下文动态生成向量，取代静态嵌入成为主流

## 来源
- [[word2vec_2013]] — 提出 CBOW 与 Skip-gram 两种高效词向量学习方法
