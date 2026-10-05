---
type: log
---
## [Update] resnet_2015 Source 页补全 + attention_mechanism 错挂摘除

- 更新 Source: [[resnet_2015]] — 按 `schema/source.md` 六栏结构补全「关键要点（5 条）/ 方法·发现（ImageNet 训练配置、层响应分析、COCO 检测增益）/ 局限·意义」，概述与元数据补入会议出处（CVPR 2016）、作者全名与中文别名
- 修正: `related_nodes` 摘除 `attention_mechanism` — 该关联无谱系依据，属早期建页占位；经核查 [[attention_mechanism]] 页的 `related_nodes` 与正文均未反向引用，属单向错挂，摘除后无残留
- 双向链接: `related_nodes` 补入 13 项反链页（microsoft / kaiming_he / [[residual_connection]] / [[bottleneck_architecture]] / [[convolutional_neural_network]] / [[batch_normalization]] / [[ioffe_2015_batchnorm]] / [[hu_2017_senet]] / [[gao_2019_res2net]] / [[ding_2021_repvgg]] / [[wang_2017_nonlocal]] / [[chen_2017_deeplabv3]] / [[zhao_2016_pspnet]]），双向性审计通过
- 更新 Concept: [[bottleneck_architecture]] — 补入 `[[resnet_2015]]` 反链与「来源」条目；该页定义本就写明「1×1 瓶颈：Inception 与 ResNet Bottleneck 的共同设计」，此前却未引用 ResNet 原文
- 数据来源: 全部取自 [raw/cnn/resnet.md](../../raw/cnn/resnet.md) §3.1–§4.3（原文中英对照笔记），未引入外部信息
- 已核查无需改动: 其余 11 个引用 `resnet_2015` 的页面 `related_nodes` 均已含反向引用；`wiki/sources/index.md` 索引行无需调整
- 遗留问题（本次不处理）: `attention_mechanism` 被当作通用占位符挂在约 90 个 Source 页上（word2vec_2013 / bpe_2015 / glide 等），语义关联不足，建议单独立项清理
