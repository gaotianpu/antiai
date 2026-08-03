---
id: contrastive_language_image_pretraining
type: concept
tags: [NLP, computer-vision, machine-learning, empirical-study]
aliases: [CLIP, 对比语言图像预训练]
related_nodes: [clip, contrastive_learning, multimodal_language_model]
last_verified: 2026-08-03
---

# Contrastive Language-Image Pretraining (CLIP)

## 定义
用 4 亿"图像-文本"对做对比预训练：拉近匹配图文对的嵌入、推远不匹配对，学到跨模态对齐表示，零样本迁移到下游视觉任务。

## 关键要点
- **对比目标**：batch 内图文对匹配的 InfoNCE 式损失
- **零样本分类**：类别名拼成文本模板，与图像算相似度即可分类
- **影响**：多模态对齐的基础模型，催生 DALL·E 2、BLIP 等系列工作

## 来源
- [[clip]] — CLIP 原始论文
