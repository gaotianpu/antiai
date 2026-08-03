---
id: weakly_supervised_speech_recognition
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [弱监督语音识别, 大规模弱监督]
related_nodes: [radford_2022_whisper, transfer_learning]
last_verified: 2026-08-03
---

# Weakly Supervised Speech Recognition

## 定义
用大规模弱标签（网络抓取的"音频-转录文本"对，无需人工精标）训练语音识别模型的范式，如 Whisper 在 68 万小时多语言数据上训练。

## 关键要点
- **数据规模**：弱监督数据量级远超人工标注语料
- **零样本迁移**：无需微调即可跨语言、跨任务迁移，达到有监督 SOTA
- **关键**：数据多样性与规模比标注精度更重要

## 来源
- [[radford_2022_whisper]] — Whisper
