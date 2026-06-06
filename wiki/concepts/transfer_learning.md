---
id: transfer_learning
type: concept
tags: [machine-learning, empirical-study, practical-guide]
aliases: [迁移学习, fine-tuning, 微调]
related_nodes: [devlin_2018_bert, he_2019_moco, chen_2020_simclr, brown_2020_gpt3, radford_2018_gpt, mae]
last_verified: 2026-06-06
---

# Transfer Learning

## 定义
Transfer Learning（迁移学习）指在一个任务（通常是大规模数据预训练）上习得的知识被迁移到另一个相关任务上的学习范式。在深度学习时代，典型实践是"预训练 + 微调"（pretrain-then-finetune）。

## 核心范式
- **特征迁移** — 冻结预训练模型提取通用特征，仅训练任务头
- **全模型微调** — 在目标任务上继续更新所有参数
- **参数高效微调** — 仅适配少量参数（如 [[low_rank_adaptation]]）
- **Prompt Tuning** — 在输入层添加可学习提示

## 关键应用
- NLP：BERT（[[devlin_2018_bert]]）、GPT（[[radford_2018_gpt]]）
- 视觉：MoCo（[[he_2019_moco]]）、SimCLR（[[chen_2020_simclr]]）、MAE（[[mae]]）
- 多模态：CLIP（[[clip]]）

## 来源
- [[devlin_2018_bert]] — NLP 预训练+微调范式奠基
- [[he_2019_moco]] — 视觉对比学习迁移
- [[chen_2020_simclr]] — SimCLR 视觉表示迁移
