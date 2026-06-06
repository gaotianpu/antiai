# 新手入门

AI 技术演进知识库，适合零基础起步。

## 整体框架

AI 模型 = 数据 + 架构 + 训练范式 + 对齐。以下按学习路径排列。

---

## 第一步：基础组件

理解 AI 模型由哪些基本零件组成。

| 概念 | 说明 | 学习文档 |
|:---|:---|:---|
| 向量与张量 | 一切数据（文本/图像/音频）最终表示为张量 | `my_notes/basic/vector.md` |
| 线性回归 | 最简单的预测模型，理解梯度下降 | `my_notes/basic/linear_regression.md` |
| 逻辑回归 | 分类问题，交叉熵损失 | `my_notes/basic/logistic_regression.md` |
| 多层感知机 | 线性 + 非线性激活的堆叠，万能近似定理 | `my_notes/basic/mlp.md` |
| 激活函数 | ReLU / Sigmoid / Tanh / GELU 等 | `my_notes/basic/Activation.md` |
| 归一化 | Batch / Layer / Group Normalization | `my_notes/basic/Normalization.md` |
| 损失函数 | MSE / 交叉熵 / 对比损失等 | `my_notes/basic/Loss_metric.md` |
| 优化器 | SGD / Adam / AdamW 等 | `my_notes/basic/Optimizer.md` |
| 正则化 | Dropout / 权重衰减 / 早停 | `my_notes/basic/regularization.md` |

## 第二步：理解架构演化

深度学习的架构经历了从简单到统一的过程。

```
MLP → RNN(LSTM/GRU) → CNN → Transformer → ?
```

- **RNN / LSTM / GRU** — 序列建模（文本、时序）
  - 论文: [[elman_1990_rnn]] → [[hochreiter_1997_lstm]] → [[gru_2014]]
- **CNN** — 图像处理（卷积、池化、残差连接）
  - 入门: `my_notes/basic/Object_Detection.md`
  - 经典论文: LeNet → AlexNet → VGG → ResNet
- **Transformer** — 统一架构，当前主流
  - 核心论文: [[vaswani_2017_transformer]]
  - 入门笔记: `my_notes/basic/intro_transformer.md`
  - 位置编码: [[roformer_2021]]（RoPE） | [[alibi_2021]]（ALiBi） | [[xpos_2022]]
  - 长序列: [[dai_2019_transformer_xl]] | [[longnet_2023]] | [[retnet_2023]]

## 第三步：理解三大训练范式

预训练范式与架构正交——同一种架构可以做不同的预训练任务。

| 范式 | 直观理解 | NLP 代表 | 视觉代表 | 多模代表 |
|:---|:---|:---|:---|:---|
| **掩码自动编码** | 完形填空 | [[devlin_2018_bert]] | [[mae]] / [[beit_2]] | [[beit_v3]] |
| **生成式自回归** | 预测下一个 token | GPT 系列 | [[igpt]] | [[kosmos_1]] |
| **对比学习** | 拉近相似、推远不同 | [[cpt_txt_2022]] | MoCo / SimCLR | [[clip]] |

初学者先理解三种范式的直观区别，再读对应论文。

- [[survey_prompting_2021]] — 预训练 + 提示方法的系统性综述
- [[radford_2018_gpt]] → [[radford_2019_gpt2]] → [[brown_2020_gpt3]] — GPT 系列演化
- [[devlin_2018_bert]] → [[liu_2019_roberta]] — BERT 及其优化

## 第四步：理解大模型的关键技术

### 4.1 缩放与涌现
随着模型规模增大，出现小模型不具备的能力（上下文学习、推理等）。
- [[brown_2020_gpt3]] — 首次系统研究 few-shot 上下文学习
- [[wei_2022_cot]] — 思维链提示激发推理能力
- [[zero_shot_cot_2022]] — 零样本思维链

### 4.2 人类对齐（RLHF）
让大模型输出符合人类期望。
- [[ouyang_2022_instructgpt]] — RLHF 三阶段流程（SFT → RM → PPO）
- [[summarize_hf_2020]] — 人类反馈用于摘要任务
- [[reinforcement_learning_from_human_feedback]] — 相关概念

### 4.3 参数高效微调
不更新全部参数，用插件式方法适配下游任务。
- `my_notes/basic/Inference_deploy.md` — 推理部署基础

### 4.4 注意力机制
- [[bahdanau_2014_attention]] — 注意力机制起源
- [[dao_2022_flashattention]] — IO 感知的高效注意力
- [[attention_mechanism]] — 概念总览

## 第五步：多模态与前沿

当文本、图像、音频等模态统一到 Transformer 架构后，多模态融合成为趋势。

- [[clip]] — 对比语言-图像预训练
- [[segment_anything]] — 分割一切（SAM）
- [[dall_e_v2]] — 文本到图像生成
- [[cheng_2026_engram]] — 条件记忆轴（最新前沿）

---

## 推荐阅读顺序

```
基础组件（第一步）→ 架构演化（第二步）→ 预训练范式（第三步）
→ Transformer 深入 → LLM 关键技术（第四步）→ 多模态（第五步）
```

每步先读 `my_notes/basic/` 下的入门笔记，再读论文摘要（`wiki/sources/`），深入时看 `raw/` 下的完整笔记。
