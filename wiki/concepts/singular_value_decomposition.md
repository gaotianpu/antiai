---
id: singular_value_decomposition
type: concept
tags: ["machine-learning", "theoretical"]
aliases: ["奇异值分解", "SVD", "奇异值"]
related_nodes: [linear_algebra, eigendecomposition, low_rank_adaptation, model_pruning]
---

# 奇异值分解

奇异值分解（SVD）将任意矩阵分解为旋转 + 缩放 + 旋转的标准形式，是线性代数中最通用的矩阵分解工具。

## 核心理论

- **分解形式**：$A_{m \times n} = U_{m \times m} \Sigma_{m \times n} V^\top_{n \times n}$
  - $U$：左奇异向量，列正交，构成输出空间的正交基
  - $\Sigma$：对角矩阵，对角元 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ 为奇异值
  - $V^\top$：右奇异向量，行正交，构成输入空间的正交基
- **低秩近似 (Eckart-Young 定理)**：保留前 $k$ 个最大奇异值，截断其余：$A \approx U_k \Sigma_k V_k^\top$。这是所有低秩分解的数学基础
- **与特征分解的关系**：$A^\top A$ 的特征值 $\lambda_i = \sigma_i^2$，右奇异向量 $v_i$ 是 $A^\top A$ 的特征向量；$AA^\top$ 给出左奇异向量 $u_i$

## DL 中的 SVD

| 应用 | 关系 |
|:---|:---|
| LoRA 低秩适配 | $W = W_0 + BA$ 等价于对适配增量做截断 SVD：$B$ 和 $A$ 分别对应 $U_k \Sigma_k^{1/2}$ 和 $\Sigma_k^{1/2} V_k^\top$ |
| 模型压缩与剪枝 | 将权重矩阵做 SVD 后截断小奇异值分量，直接减少参数量和计算量；本质是 [[model_pruning]] 的一种结构化形式 |
| PCA | 对数据矩阵做 SVD 即可得到主成分方向（$V$ 的列），比协方差矩阵特征分解数值更稳定 |
| 伪逆计算 | 矩阵的 Moore-Penrose 伪逆 $A^+ = V \Sigma^+ U^\top$，用于线性最小二乘求解 |
