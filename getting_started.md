# 新手入门

从零搭建 AI 模型的理解，每次只加一个新概念。

---

## Step 0：最小可运行

```python
# 一个输入 → 一个输出，没有激活函数，没有层
y = w * x + b
```

- **代码**: `my_notes/basic/data/linear.py` — 用梯度下降学出一条直线
- **概念**: [[word2vec_2013]] 词向量？不是，先理解标量、向量 | `my_notes/basic/vector.md`
- **论文**: [[elman_1990_rnn]]？太早了，先跑通线性回归

---

## Step 1：加一个非线性

```python
# 线性 + Sigmoid = 逻辑回归，输出变成概率
p = sigmoid(w * x + b)
```

- **代码**: `my_notes/basic/data/logistic.py`
- **新概念**: 激活函数、交叉熵损失
- **文档**: `my_notes/basic/Activation.md` | `my_notes/basic/Loss_metric.md`
- **论文**: 先不读论文，跑通代码再说

---

## Step 2：堆叠成网络

```python
# 多个神经元堆叠 = 多层感知机
h = relu(W1 @ x + b1)
y = W2 @ h + b2
```

- **代码**: `my_notes/basic/data/xor.py` — 经典异或问题，单层线性无解，加一层就解了
- **新概念**: 隐藏层、反向传播、梯度消失
- **文档**: `my_notes/basic/mlp.md`
- **论文**: 可以先读 [[hochreiter_1997_lstm]] 的梯度问题部分

---

## Step 3：让它 train 得更好

```python
# 加归一化 → 加正则化 → 换优化器
h = layer_norm(relu(W1 @ x + b1))
y = W2 @ dropout(h) + b2  # AdamW 优化
```

- **文档**: `my_notes/basic/Normalization.md` | `my_notes/basic/regularization.md` | `my_notes/basic/Optimizer.md`
- **论文**: [[layer_norm_2016]] | [[liu_2019_roberta]]（RoBERTa 展示了优化细节的影响）

---

## Step 4：换一种架构 — CNN

```python
# 图像不用全连接，用卷积（参数共享 + 局部连接）
h = conv2d(x, kernel)      # 卷积层
p = max_pool2d(h)          # 池化层
y = flatten(p) @ W + b     # 分类头
```

- **文档**: `my_notes/basic/Object_Detection.md`
- **论文**: 从 LeNet 读到 ResNet
- **关键突破**: 残差连接让深层网络可训练 → [[vaswani_2017_transformer]] 借用了这个思想

---

## Step 5：换一种架构 — RNN

```python
# 序列数据：当前输出依赖上一时刻的隐藏状态
h_t = tanh(W_h @ h_{t-1} + W_x @ x_t)
y_t = W_y @ h_t
```

- **论文**: [[elman_1990_rnn]] → [[hochreiter_1997_lstm]] → [[gru_2014]]
- **问题**: 长序列梯度消失 → LSTM 的门控机制

---

## Step 6：核心突破 — Attention

```python
# 不再把输入压缩成一个向量，而是每一步都可以回头看
# 注意力 = 查询(Q) 与 键(K) 的相似度加权 值(V)
scores = Q @ K.T / sqrt(d)
weights = softmax(scores)
output = weights @ V
```

- **概念**: [[attention_mechanism]]
- **论文**: [[bahdanau_2014_attention]] → [[vaswani_2017_transformer]]
- **代码**: 先用手写 np 实现单头注意力，再看多头
- **关键**: 去掉 RNN，全部用注意力 → Transformer 诞生

---

## Step 7：Transformer 及其变体

```python
# Transformer 块 = 多头注意力 + 残差 + FFN + LayerNorm
def transformer_block(x):
    x = x + multi_head_attn(layer_norm(x))
    x = x + ffn(layer_norm(x))
    return x
```

- **文档**: `my_notes/basic/intro_transformer.md`
- **位置编码**: [[roformer_2021]]（RoPE）| [[alibi_2021]] | [[xpos_2022]]
- **高效注意力**: [[dao_2022_flashattention]]
- **超长序列**: [[dai_2019_transformer_xl]] | [[longnet_2023]]
- **视觉版**: [[vit]] — 图像切成 patch，当成序列

---

## Step 8：预训练范式

同一个 Transformer 架构，换不同的训练目标：

```
掩码（BERT）→ 自回归（GPT）→ 对比学习（CLIP）
```

- **BERT**: [[devlin_2018_bert]] → [[liu_2019_roberta]]
- **GPT**: [[radford_2018_gpt]] → [[radford_2019_gpt2]] → [[brown_2020_gpt3]]
- **对比学习**: [[clip]] | [[cpt_txt_2022]]
- **综述**: [[survey_prompting_2021]] | [[survey_transformers_2021]]

---

## Step 9：大模型的关键能力

### 上下文学习
- [[brown_2020_gpt3]] — 给几个示例就能做新任务，不更新参数
- [[in_context_learning]] — 概念详解

### 思维链推理
- [[wei_2022_cot]] — 让模型输出中间步骤
- [[auto_cot_2022]] — 自动生成 CoT 示例
- [[zero_shot_cot_2022]] — "Let's think step by step"

### 人类对齐
- [[ouyang_2022_instructgpt]] — RLHF 三阶段
- [[reinforcement_learning_from_human_feedback]] — 概念详解

---

## Step 10：多模态

- [[clip]] — 文本 + 图像对比学习
- [[segment_anything]] — 分割一切
- [[dall_e_v2]] — 文本 → 图像
- [[cheng_2026_engram]] — 条件记忆（最新前沿）

---

## 代码优先路线

```
Step 0: my_notes/basic/data/linear.py        ← 先跑起来
Step 1: my_notes/basic/data/logistic.py       ← 加非线性
Step 2: my_notes/basic/data/xor.py            ← 堆叠成网络
Step 3: my_notes/basic/data/iris_dl.py        ← 完整训练流程
Step 4: 手写 Attention (NumPy)                ← 理解核心机制
Step 5: 读 transformers 库源码                ← 看工业实现
```

每步先跑代码，再读论文，最后读 wiki 概念页。
