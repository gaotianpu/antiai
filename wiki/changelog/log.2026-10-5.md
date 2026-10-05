---
type: log
---
## [Update] resnet_2015 Source 页补全 + attention_mechanism 错挂摘除

- 更新 Source: [[resnet_2015]] — 按 `schema/source.md` 六栏结构补全「关键要点（5 条）/ 方法·发现（ImageNet 训练配置、层响应分析、COCO 检测增益）/ 局限·意义」，概述与元数据补入会议出处（CVPR 2016）、作者全名与中文别名
- 修正: `related_nodes` 摘除 `attention_mechanism` — 该关联无谱系依据，属早期建页占位；经核查 [[attention_mechanism]] 页的 `related_nodes` 与正文均未反向引用，属单向错挂，摘除后无残留
- 双向链接: `related_nodes` 曾误补入 13 项反链页（含 12 个「引用本页」的页面），**同日已回退**——理由见下方修正条目
- 更新 Concept: [[bottleneck_architecture]] — 补入 `[[resnet_2015]]` 反链与「来源」条目；该页定义本就写明「1×1 瓶颈：Inception 与 ResNet Bottleneck 的共同设计」，此前却未引用 ResNet 原文
- 数据来源: 全部取自 [raw/cnn/resnet.md](../../raw/cnn/resnet.md) §3.1–§4.3（原文中英对照笔记），未引入外部信息
- 已核查无需改动: 其余 11 个引用 `resnet_2015` 的页面 `related_nodes` 均已含反向引用；`wiki/sources/index.md` 索引行无需调整
- 遗留问题（本次不处理）: `attention_mechanism` 被当作通用占位符挂载于 100 个页面（word2vec_2013 / bpe_2015 / elman_1990_rnn 等），语义关联不足，另行立项清理

## [Fix] resnet_2015 related_nodes 过度平铺回退 2026-10-05

- 修正: [[resnet_2015]] 的 `related_nodes` 由 13 项收敛至 6 项（microsoft / kaiming_he / [[residual_connection]] / [[bottleneck_architecture]] / [[convolutional_neural_network]] / [[batch_normalization]]）
- 依据: [[best_practices]] §12「related_nodes 应有方向性，而非平铺所有相关链接」+ `schema/frontmatter.md` 对 related_nodes 的定义（**本页引用**的页面列表）——二者均不支持把「引用本页」的页面反向平铺进来；resnet_2015 被 12 页引用，属中心页，`wiki-lint` 检查 6 明确豁免中心页的反向平铺
- 保留项依据: 6 项均在本页元数据/正文中显式引用（机构、作者、新颖概念、方法段、引用段）；回退后 A→X 方向的反向引用仍完整（6 个目标页的 `related_nodes` 均含 resnet_2015）
- 教训: 「双向性」的规范方向是「**本页引用的页面须反链本页**」，不是「引用本页的页面须被本页列出」——后者会随中心页热度线性膨胀

