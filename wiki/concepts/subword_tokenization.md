---
id: subword_tokenization
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [子词分词, 子词切分]
related_nodes: [bpe_2015, byte_pair_encoding, tokenization]
last_verified: 2026-08-03
---

# Subword Tokenization

## 定义
介于词级与字符级之间的分词粒度：将词拆分为子词单元，平衡词表规模与语义完整性。

## 关键变体
| 方法 | 特点 | 代表模型 |
|:---|:---|:---|
| BPE | 按频率合并符号对 | GPT、RoBERTa |
| WordPiece | 按似然增益合并 | BERT |
| Unigram | 按概率最大化剪枝 | T5、XLNet |
| SentencePiece | 语言无关的通用框架 | T5、LLaMA |

## 来源
- [[bpe_2015]] — 子词分词的开创性工作
