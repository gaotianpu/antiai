---
id: linear_algebra
type: concept
tags: [theoretical, machine-learning]
aliases: ["线性代数", "矩阵运算", "linear algebra for DL"]
related_nodes: [math_for_deep_learning, optimization_fundamentals, matrix_operations, eigendecomposition, singular_value_decomposition]
---

# 线性代数

线性代数是深度学习最直接的数学工具，神经网络的每一层本质上是线性变换加非线性激活。

## 核心概念

- **向量与矩阵**：数据的基本表示形式。标量 → 向量 → 矩阵 → 张量构成 DL 的数据层级
- **矩阵乘法**：全连接层 $y = Wx + b$、注意力 $\text{Attention}(Q,K,V) = \text{softmax}(QK^\top)V$ 的核心运算
- **转置与逆**：矩阵转置 $A^\top$ 改变维度排列；逆矩阵 $A^{-1}$ 用于求解线性系统，但在 DL 中因规模巨大很少直接计算
- **特征值与特征向量**：满足 $Av = \lambda v$，用于理解线性变换的缩放行为，是 PCA 和谱分析的基础
- **奇异值分解 (SVD)**：$A = U\Sigma V^\top$，将任意矩阵分解为旋转+缩放+旋转。用于低秩近似、降维、伪逆计算
- **范数 (Norm)**：
  - L1 范数：$\|x\|_1 = \sum|x_i|$，诱导稀疏性（Lasso 正则化）
  - L2 范数：$\|x\|_2 = \sqrt{\sum x_i^2}$，衡量向量长度（权重衰减）
- **矩阵微积分 (Matrix Calculus)**：对标量函数关于矩阵求导，是梯度推导的形式化工具

## 为什么 DL 需要它

| 应用 | 线性代数基础 |
|:---|:---|
| 权重矩阵 $W$ | 线性变换的参数载体 |
| Embedding 层 | 查表运算等价于稀疏矩阵乘法 |
| 注意力计算 | $QK^\top$ 矩阵乘法 + softmax 归一化 |
| LoRA 微调 | $W = W_0 + BA$，利用低秩分解减少参数量 |
| 卷积运算 | 可重写为 Toeplitz 矩阵乘法 |
| 反向传播 | 每一层梯度是 Jacobian 矩阵的链式乘积 |
