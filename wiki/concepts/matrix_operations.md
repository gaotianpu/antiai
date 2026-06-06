---
id: matrix_operations
type: concept
tags: ["machine-learning", "theoretical"]
aliases: ["矩阵运算", "矩阵乘法", "矩阵分解", "矩阵"]
related_nodes: [linear_algebra, low_rank_adaptation]
---

# 矩阵运算

矩阵运算构成深度学习计算的基本操作，从数据表示到模型推理无处不在。

## 核心运算

- **矩阵乘法**：$C = AB$ 是最常见操作。注意力机制的核心 $QK^\top$ 即为矩阵乘法，计算所有 query-key 对的相似度
- **转置 (Transpose)**：$A^\top$ 交换行列维度。注意力计算中 $K^\top$ 使 key 的维度对齐 query
- **逆矩阵 (Inverse)**：$A^{-1}$ 满足 $AA^{-1}=I$，用于线性系统求解；DL 中因计算复杂很少直接用
- **范数 (Norm)**：
  - L1 范数：$\|x\|_1 = \sum|x_i|$ — 诱导稀疏性，用于 Lasso 正则化
  - L2 范数：$\|x\|_2 = \sqrt{\sum x_i^2}$ — 欧几里得距离，权重衰减的几何基础
  - Frobenius 范数：$\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$ — 矩阵版本的 L2 范数
- **迹 (Trace)**：$\operatorname{tr}(A) = \sum_i a_{ii}$，对角元之和。用于矩阵微积分推导，如 $\frac{\partial \operatorname{tr}(AB)}{\partial A} = B^\top$
- **秩 (Rank)**：矩阵列/行向量的最大线性无关数。低秩矩阵可用更少参数表示（见 [[low_rank_adaptation]]）

## DL 中的矩阵运算

| 应用 | 涉及操作 |
|:---|:---|
| 权重矩阵 $W$ | 线性变换 $y = Wx$ 的本质是矩阵乘法 |
| Embedding 层 | 查表等价于稀疏矩阵与 one-hot 向量的乘法 |
| 注意力计算 $QK^\top$ | 矩阵乘法 + softmax 归一化 |
| LoRA | $W_0 + BA$，$B, A$ 为低秩矩阵，乘积接近原矩阵秩 |
