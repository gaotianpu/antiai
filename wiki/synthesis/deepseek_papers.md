---
id: deepseek_papers
type: synthesis
tags: ["survey", "NLP", "machine-learning"]
aliases: ["DeepSeek 论文索引", "DeepSeek papers overview", "深度求索论文清单"]
related_nodes: [deepseek_ai, mixture_of_experts, conditional_memory]
---

# DeepSeek 论文索引

[!ABSTRACT] [[deepseek_ai|DeepSeek]]（深度求索）AI 研究公司的核心论文与技术报告汇总。其技术贡献主要通过 arXiv 预印本、GitHub 技术报告以及学术期刊/会议发布。

---

## 核心模型系列

| 模型 | 核心贡献 | 引用 |
|:---|:---|:---|
| DeepSeek LLM (67B) | Scaling Laws、超参数选择、数据质量影响 | *arXiv:2401.02954* |
| DeepSeek-V2 | 236B [[mixture_of_experts|MoE]] + Multi-head Latent Attention，KV cache 压缩 93.3% | *arXiv:2405.04434* |
| DeepSeek-V3 | 671B MoE (37B 活跃)，无辅助损失负载均衡 + 多 Token 预测训练 | *arXiv:2412.19437* |
| DeepSeek-V3.1 | Think/非 Think 混合推理模型 + Agent | — |
| DeepSeek-V3.2 | 稀疏注意力 DSA + [[reinforcement_learning|RL]] 提升效率与推理 | DeepSeek-AI (2025) |
| DeepSeek-R1 | 纯 RL 涌现推理能力，Nature 封面报道 | *arXiv:2501.12948* |
| DeepSeek-Prover-V2 | RL 推进形式化数学推理 | DeepSeek-AI (2025) |

## 前沿架构创新

- **[[conditional_memory|Engram（条件记忆）]]**：将模型"记忆"与"计算"能力解耦的全新神经网络模块
- **mHC（流形约束超连接）**：通过信息"智能调节阀"提升大规模训练稳定性

## 专项领域模型

| 模型 | 领域 | 引用 |
|:---|:---|:---|
| DeepSeek-OCR | 视觉压缩：长文本光学二维映射 | *arXiv:2510.18234* |
| Thinking with Visual Primitives | 多模态推理：坐标等视觉原语融入 [[chain_of_thought|CoT]] | — |
| DeepSeekMath-V2 | 数学推理：自验证能力，竞赛级 | — |
| DeepSeek-Coder | 代码智能 | *arXiv:2401.14196* |
| DeepSeek-VL | 多模态 | — |

## 底层系统与效率

- **FlashMLA**：双向管道并行，压缩流水线气泡、共享梯度传输，为 DeepSeek-V3/R1 设计
- **Native Sparse Attention (NSA)**：原生稀疏注意力，长上下文建模效率

## 辅助资源

- DeepSeek-V3 技术报告深度解析（架构创新与工程实践）
- DeepSeek 系列论文技术要点 PPT（60 页）
- DeepSeek 进化史（基于 21 篇论文的技术蜕变）

---

## 关联页面

- [[deepseek_ai]] — 研究机构实体
- [[mixture_of_experts]] — MoE 稀疏激活范式
- [[conditional_memory]] — Engram 条件记忆架构
