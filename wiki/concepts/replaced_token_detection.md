---
id: replaced_token_detection
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [替换词检测, RTD]
related_nodes: [electra_2020_electra, masked_language_modeling]
last_verified: 2026-08-03
---

# Replaced Token Detection (RTD)

## 定义
ELECTRA 提出的预训练目标：生成器替换部分 token 为 plausible 的假词，判别器判断每个位置是否被替换，计算效率远高于 MLM。

## 关键要点
- **判别任务**：对所有位置二分类（真/假），学习信号密度高
- **效率优势**：同等算力下比 BERT/MLM 收敛更快、效果更好
- **GAN 式结构**：生成器-判别器联合训练，但使用最大似然而非对抗

## 来源
- [[electra_2020_electra]] — ELECTRA 原始论文
