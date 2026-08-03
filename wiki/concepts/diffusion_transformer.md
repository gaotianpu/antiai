---
id: diffusion_transformer
type: concept
tags: [machine-learning, computer-vision, empirical-study]
aliases: [扩散Transformer, DiT]
related_nodes: [dits, diffusion_model, transformer_architecture]
last_verified: 2026-08-03
---

# Diffusion Transformer (DiT)

## 定义
以 Transformer（而非 U-Net）作为扩散模型骨干的架构：将图像 patch 化为 token 序列，用 DiT 块替代 U-Net 的卷积块，扩展性更强。

## 关键要点
- **替换骨干**：U-Net 的归纳偏置 → Transformer 的通用性与可扩展性
- **条件注入**：adaLN（自适应层归一化）融合类别/文本条件
- **意义**：Sora 等视频生成、现代文生图模型的基础架构

## 来源
- [[dits]] — DiT：扩散 Transformer
