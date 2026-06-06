# 代码实践

## 一、基础组件
### 1. Tokenizer
https://pytorch.org/text/stable/transforms.html?highlight=Tokenizer

1. GPT2BPE  https://github.com/openai/gpt-2/blob/master/src/encoder.py
    * https://github.com/openai/tiktoken Rust实现，效率很高
2. CLIP(Byte-Level BPE) https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/tokenizer.py
    将空格视为标记的一部分（有点像句子片段），因此无论单词是否在句子开头（没有空格），都会以不同的方式编码
3. SentencePiece https://github.com/google/sentencepiece
4. BERT(WordPiece)  
5. Regex https://github.com/google/re2

#### 特殊的tokens
* padding
* mask
* eos

### 2. Embedding 嵌入
嵌入类似矩阵，每行是一个向量，每列是向量某个维度的值，比矩阵多了一个行编号，可以根据编号获取指定向量信息。

https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
两个主要参数：大小num_embeddings， 每个向量的维度 embedding_dim
padding_idx 指定某个词向量不更新梯度.

#### 2.1 Token Embedding 词嵌入

#### 2.2 Position Embedding 位置嵌入
* sin/cos
* learnable 可学习的词嵌入
* [RoPE](#)
* [ALibi](../paper/nlp/Alibi.md)
* [XPos](../paper/nlp/XPOS.md)

## 3. self-Attention
https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

实例化阶段：模型的维度，
forward: q,k,v; key_padding_mask 输入长度不到最大长度时; attn_mask； is_causal 因果掩码；

### BCA
分块因果注意力(blockwise causal attention)

### 4. LayerNorm
https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

使用的时机: 
* attention后，FFN后, 原始论文中提到; 
* attention前: SubLN,DeepNorm

## 5. FFN and Router
稀疏专家？ Router和FFN

## 6. 初始化方法

## 二、高级组合
### 1. Transformer
* https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html

### 2. TransformerEncoder
https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html

[BERT: Bidirectional Encoder Representations from Transformers](../paper/nlp/bert.md)

## 3. TransformerDecoder
* https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html
* https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html


[GPT](../paper/nlp/gpt.md)
norm_first， 微软 [MAGNETO](../paper/Multimodal/MAGNETO.md) sub-LN还没没实现

