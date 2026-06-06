---
id: knowledge_distillation
type: concept
tags: [machine-learning, empirical-study, parameter-optimization]
aliases: [蒸馏, KD, 知识蒸馏]
related_nodes: [hsieh_2023_distilling, han_2015_deepcompression, tinybert_2019_tinybert, mobilebert_2020_mobilebert]
last_verified: 2026-06-06
---

# Knowledge Distillation

## 定义
Knowledge Distillation（知识蒸馏）是一种模型压缩技术：用大型教师模型（teacher）的软标签（soft targets）或中间表示来监督小型学生模型（student）的训练，使学生模型逼近教师性能的同时保持更低的计算成本。

## 核心方法
- **软标签蒸馏**（Hinton 2015）— 使用温度参数软化教师输出概率分布
- **特征蒸馏** — 对齐教师与学生中间层的特征表示
- **关系蒸馏** — 保留样本间的结构关系（如 pairwise similarity）
- **自蒸馏** — 模型自身作为教师（如 [[tinybert_2019_tinybert]]）

## 来源
- [[hsieh_2023_distilling]] — Distilling Step-by-Step 少步蒸馏
- [[han_2015_deepcompression]] — 蒸馏+剪枝+量化的联合压缩框架
- [[tinybert_2019_tinybert]] — BERT 蒸馏的代表工作
- [[mobilebert_2020_mobilebert]] — 面向移动端的蒸馏架构
