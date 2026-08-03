---
id: dual_encoder
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [双编码器, 孪生编码器]
related_nodes: [dual_encoder_qa_2022, siamese_network, contrastive_learning]
last_verified: 2026-08-03
---

# Dual Encoder

## 定义
用两个编码器分别编码查询与文档（可共享或独立参数），在共享嵌入空间计算相似度的架构，检索类任务的标配。

## 关键要点
- **独立编码**：查询与文档可离线预计算，检索时仅算相似度
- **与交互式编码器对比**：双编码器牺牲细粒度交互换效率
- **训练**：对比学习（负采样）是标准训练方式

## 来源
- [[dual_encoder_qa_2022]] — 双编码器问答架构研究
