---
id: tokenization
type: concept
tags: [NLP, machine-learning]
aliases: [分词, tokenizer, subword]
related_nodes: [attention_mechanism]
---

# Tokenization

## 概述

Tokenization（分词）是将自然语言文本切分为离散单元（token）的过程，是 NLP 系统处理文本的第一环节。分词策略直接影响词表大小、序列长度和模型表现。

## 详细阐述

### 定义与内涵

Tokenization 将输入文本映射为 token ID 序列。按粒度分为三级：**字/字符级**（Char-level）、**词级**（Word-level）和 **子词级**（Subword-level）。现代 NLP 系统以子词分词为主流。

### 主要算法

#### BPE（Byte Pair Encoding）

从字符集合开始，迭代合并最频繁的相邻 token 对，直到达到目标词表大小 [[bpe_2015]]。GPT、BART 等模型使用。

#### WordPiece

类似 BPE，但合并标准基于互信息（互信息最大的一对合并）。BERT [[bert_wwm_2019]]、ELECTRA [[electra_2020_electra]] 使用。

#### SentencePiece

Google 开源的端到端分词库 [[raffel_2019_t5]]。将原始文本视为 Unicode 字符序列，支持 BPE 和 Unigram LM 两种模式，无需预分词。

#### Unigram LM

基于概率语言模型的分词方法。通过 EM 算法迭代删除低概率 token，剩余集合形成词表。XLNet [[xlnet_2019]]、T5 [[raffel_2019_t5]] 使用。

### 关键维度

| 维度 | BPE | WordPiece | Unigram LM | SentencePiece |
|:---|:---|:---|:---|:---|
| 合并策略 | 频率 | 互信息 | 概率似然 | BPE 或 Unigram |
| 预分词需求 | 是 | 是 | 是 | 否 |
| 确定性 | 是 | 是 | 否（采样） | 取决于模式 |
| 典型用户 | GPT 系列 | BERT | XLNet, T5 | T5, mT5 |

### 常见问题

- **OOV（Out-of-Vocabulary）**：子词分词基本解决，生僻词退化为字符
- **词表大小权衡**：大词表 → 短序列但稀疏；小词表 → 长序列但密集
- **多语言公平性**：SentencePiece 对非拉丁语系更友好

## 相关概念网络

- [[attention_mechanism]] — 自注意力的输入依赖分词结果

## 引用资料

1. [[bpe_2015]] — BPE 分词算法
2. [[bert_wwm_2019]] — WordPiece 分词的典型应用
3. [[raffel_2019_t5]] — SentencePiece + Unigram LM
4. [[xlnet_2019]] — Unigram LM 的应用

## 更新记录

- **2026-06-06**: 初始创建
