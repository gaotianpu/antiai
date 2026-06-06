---
id: low_rank_adaptation
type: concept
tags: [machine-learning, parameter-optimization, empirical-study]
aliases: [LoRA, 低秩适配]
related_nodes: [hu_2021_lora, dettmers_2023_qlora, zhang_2023_adalora]
last_verified: 2026-06-06
---

# Low-Rank Adaptation (LoRA)

## 定义
Low-Rank Adaptation（LoRA）是一种参数高效微调（PEFT）方法：将预训练权重矩阵的更新量 $\Delta W$ 分解为两个低秩矩阵 $A$ 和 $B$ 的乘积，仅训练低秩矩阵即可适配下游任务，大幅减少可训练参数。

## 核心公式
$$W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d, k)$$

## 优势
- **存储友好**：为每个任务仅保存少量低秩权重（数 MB 级别）
- **部署灵活**：基座模型不变，通过插拔适配器切换任务
- **零推理延迟**：训练后可将 BA 合并到 W 中

## 衍生产品
- **QLoRA**（[[dettmers_2023_qlora]]）— LoRA + 4-bit 量化，单 GPU 微调 65B 模型
- **AdaLoRA**（[[zhang_2023_adalora]]）— 自适应分配秩参数预算

## 来源
- [[hu_2021_lora]] — LoRA 的首次提出
- [[dettmers_2023_qlora]] — 量化 LoRA 微调
- [[zhang_2023_adalora]] — 自适应低秩适配
