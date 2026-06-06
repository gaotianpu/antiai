---
id: eigendecomposition
type: concept
tags: ["machine-learning", "theoretical"]
aliases: ["特征分解", "特征值", "特征向量", "eigenvalue", "eigenvector"]
related_nodes: [linear_algebra, singular_value_decomposition]
---

# 特征分解

特征分解揭示方阵在线性变换下的不变方向，是理解矩阵本质性质的窗口。

## 核心理论

- **特征值方程**：$A v = \lambda v$，其中 $\lambda$ 为特征值，$v$ 为特征向量。矩阵 $A$ 作用于 $v$ 仅改变长度不改变方向
- **特征多项式**：$\det(A - \lambda I) = 0$，阶数等于矩阵维度，根的代数重数对应特征值的重数
- **对角化**：若 $A$ 有 $n$ 个线性无关特征向量，则 $A = PDP^{-1}$，其中 $D$ 为对角矩阵。对角化可高效计算 $A^k$
- **正定性**：对称矩阵 $A$ 正定 $\iff$ 所有特征值 $> 0$；半正定 $\iff$ 所有特征值 $\geq 0$。正定性保证优化问题的凸性
- **谱定理**：实对称矩阵可被正交对角化：$A = Q\Lambda Q^\top$，$Q$ 为正交矩阵

## DL 中的特征分解

| 应用 | 关系 |
|:---|:---|
| PCA 降维 | 对协方差矩阵做特征分解，取最大特征值对应的特征向量构成投影方向 |
| 图神经网络 (GNN) | 图拉普拉斯矩阵的谱分解定义谱域图卷积：$L = D - A$，其特征向量构成图傅里叶基 |
| Xavier 初始化 | 通过分析前向/反向传播中激活的方差，要求权重矩阵的奇异值/特征值分布在 1 附近 |
| Hessian 分析 | 损失函数 Hessian 矩阵的特征值分布刻画收敛行为：负特征值 ⇒ 鞍点；条件数 = $\lambda_{\max} / \lambda_{\min}$ 越大收敛越慢 |
