---
id: byte_pair_encoding
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [BPE, 字节对编码]
related_nodes: [bpe_2015, subword_tokenization, tokenization]
last_verified: 2026-08-03
---

# Byte Pair Encoding (BPE)

## 定义
一种数据驱动的子词分词算法：从字符/字节级符号出发，反复合并语料中出现频率最高的符号对，直到达到预设词表大小。

## 关键要点
- **解决 OOV**：任意词都可分解为子词序列，消除未登录词
- **词表可控**：通过合并轮数精确控制词表规模
- **主流地位**：GPT、BART 等预训练模型的标准分词方案

## 来源
- [[bpe_2015]] — 首次将 BPE 引入神经机器翻译解决稀有词问题
