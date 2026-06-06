# 新手入门

## 这是什么？

AI 论文知识库，持续收集和学习 AI 领域的重要论文。

## 目录结构

```
root/
├── raw/             原始论文（Markdown + PDF）
│   ├── nlp/         NLP 论文
│   ├── Multimodal/  多模态论文
│   ├── vit/         视觉 Transformer 论文
│   ├── cnn/         CNN 论文
│   └── pdf/         PDF 原文
├── wiki/            知识库
│   ├── sources/     论文摘要卡片
│   ├── concepts/    核心概念定义
│   ├── entities/    人物与组织名片
│   ├── synthesis/   跨领域分析与总结
│   └── index.md     总入口
├── schema/          操作规范
├── tools/           辅助工具
└── frontier.md      最新前沿（最近 365 天）
```

## 快速导航

- [[frontier|最新前沿]] — 最近一年的前沿论文
- [[wiki/sources/index|全部论文索引]] — 按年份浏览
- [[wiki/concepts/index|概念索引]] — 核心 AI 概念
- [[wiki/entities/index|实体索引]] — 机构与人物

## 如何贡献？

1. 将论文原文本或 PDF 放入 `raw/` 对应子目录
2. 在 `wiki/sources/` 创建摘要卡片（模板见 `schema/source.md`）
3. 提取新概念到 `wiki/concepts/`，新实体到 `wiki/entities/`
4. 更新对应 `index.md` 并提交
