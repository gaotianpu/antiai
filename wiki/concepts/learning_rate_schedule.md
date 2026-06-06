---
id: learning_rate_schedule
type: concept
tags: [machine-learning, theoretical, parameter-optimization]
aliases: [学习率策略, LR schedule, warmup, cosine decay]
related_nodes: [optimizer]
---

# Learning Rate Schedule

## 概述

Learning Rate Schedule（学习率策略）定义了训练过程中学习率随迭代次数或 epoch 的变化方式。作为 [[optimizer]] 的超参数调度组件，学习率策略直接影响收敛速度、最终性能和训练稳定性。

## 详细阐述

### 定义与内涵

学习率（learning rate）控制参数更新步长。固定学习率往往效果不佳——太大导致震荡，太小收敛过慢。学习率策略通过动态调整学习率来平衡前期快速探索和后期精细收敛。

### 主流策略

| 策略 | 模式 | 适用场景 |
|:---|:---|:---|
| Step Decay | 每隔固定 epoch 乘以衰减因子 | 传统 CNN 训练 |
| Cosine Annealing | 按余弦曲线从初始值降到 0 | Transformer、ViT |
| Linear Warmup + Cosine Decay | 先线性上升再余弦下降 | 大 batch / 大模型训练的标配 |
| Inverse Sqrt | 按 $1/\sqrt{t}$ 衰减 | 优化理论导向的方法 |
| Polynomial Decay | 按多项式函数衰减 | 分割任务（如 DeepLab） |
| OneCycle | 先升后降的单周期 | 快速训练（fast.ai） |

### Warmup 的必要性

大模型训练初期，模型参数随机初始化，统计量不稳定，大学习率可能导致梯度爆炸。Warmup 阶段从极小学习率线性增加到目标值，让优化器逐步适应。现代 Transformer（BERT、GPT）和大型视觉模型（ViT）均使用 Warmup。

### Cosine Annealing 详解

Cosine Annealing 按半周期余弦 $ \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi)) $ 调整学习率。优点是下降平滑、在接近结束时仍保持一定探索能力。常与 Warmup 结合为 Warmup-Cosine 策略，成为 Transformer 训练的事实标准。

### 与优化器的关系

不同 [[optimizer]] 对学习率调度的敏感度不同：
- SGD + Momentum：需要精细的 Step Decay 或 Cosine
- Adam / AdamW：对学习率相对不敏感，但 Warmup 仍然重要
- SGD 的 Weight Decay 实现受 LR 影响（AdamW 解耦后解决此问题）

## 相关概念网络

- [[optimizer]] — 学习率调度与优化器协同工作
- [[regularization]] — 学习率调度的正则化效应（早停等）

## 更新记录

- **2026-06-06**: 初始创建
