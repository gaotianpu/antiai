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

## [Refactor] attention_mechanism 模板占位清理（100 → 38 页）

- 问题判定: `attention_mechanism` 被 100 页引用，为第二名 `transformer_architecture`(21) 的 **5 倍**；87/100 页 `last_verified` 同为 2026-06-06（同批模板建页）；100 页中**仅 3 页**同时引用 `transformer_architecture`——三点交叉指向「模板默认占位」而非真实 hub 聚集
- 逐页分级: A 强关联 32 / B 以 Transformer 为骨干但研究别处 54 / C 无关 14；subagent 逐页读正文判定，抽查后修正 2 处边界（`deepnet`、`raffel_2019_t5` 由 A/C 归入 B）
- C 类 14 页（与注意力无关）: 摘除并替换为真实关联——`elman_1990_rnn`→[[recurrent_neural_network]]、`bpe_2015`→[[byte_pair_encoding]]/[[subword_tokenization]]、`word2vec_2013`→[[word_embedding]]/[[skip_gram]]/[[cbow]]、`hochreiter_1997_lstm`→[[long_short_term_memory]]、`sppnet_2014`→[[spatial_pyramid_pooling]] 等
- B 类 54 页（预训练策略/蒸馏/多模态系统/对齐推理/评测）: 摘除，改挂 [[transformer_architecture]] + 主题相关概念——视觉自监督系→[[vision_transformer]]、生成模型系→[[diffusion_transformer]]、BERT 变体系→[[devlin_2018_bert]]、思维链系→[[chain_of_thought]]、蒸馏系→[[knowledge_distillation]]
- 补漏收窄: 「引 `transformer_architecture` 却未引 `attention_mechanism`」的候选共 17 页，**仅 6 页应补**（bert / gpt / vision_transformer / diffusion_transformer / retention_mechanism / encoder_decoder_architecture）；其余为 [[normalization]]、[[feed_forward_network]]、[[residual_connection]]、[[speculative_decoding]] 等**并列或无关组件**，补挂会制造同类错误关联，故不补
- 正文一致性: 7 个 A 类页面（[[channel_attention]]、[[multi_head_attention]]、[[multi_head_latent_attention]]、[[non_local_operation]]、[[self_attention]]、[[kwon_2023_pagedattention]]、[[xiao_2023_streamingllm]]）此前「frontmatter 已挂但正文无链接」，补齐正文链接
- 执行与校验: 脚本批量改写 **81** 个文件；三项校验通过——68 页零残留、全库零死链、frontmatter 格式统一
- 结果: `attention_mechanism` **100 → 38**；`transformer_architecture` **21 → 73**，两 hub 职责分离（研究注意力 vs 使用 Transformer），纠正 5 倍倒挂
- 依据: [[best_practices]] §12「related_nodes 应有方向性，而非平铺所有相关链接」；`wiki-lint` 检查 6 中心页豁免

## [Lint] 第二梯队 hub 体检：无第二个模板占位

- 方法: 对引用数 ≥6 的全部 hub 统计「引用者领域分布」，检验是否存在与 `attention_mechanism` 同类的模板占位
- 结果（均健康，引用者领域高度集中）: [[iou_loss]] 11 页全为目标检测、[[sppnet_2014]] 7 页全为 CV 分割检测、[[optimization_fundamentals]] 7 页全为数学基础、[[multi_head_latent_attention]] 9 页为 DeepSeek 系 + 注意力变体、[[mixture_of_experts]] 11 页为 MoE 专门工作、[[conditional_memory]] 7 页为记忆机制聚集
- 判据修正: 「引用者 `last_verified` 集中在同一天」**无效**——整库多建于同日，且本次整理自身制造了集中（`transformer_architecture` 74%、`vision_transformer` 85%）；有效判据是**频次断崖 + 领域发散度**
- 结论: `attention_mechanism` 为孤例，`related_nodes` 体系整体健康，无需全库普查

## [Docs] 沉淀「模板占位关联」识别与清理方法至 best_practices

- 新增 Schema: `schema/best_practices.md` §15「模板占位关联的识别与清理」——沉淀本次清理的可复用判据与执行要点
- 内容: 三点交叉判据（频次断崖 / 交叉引用异常 / 领域发散）、无效判据排除、A/B/C 分级处置、三处同步与替换而非删除、补漏收窄原则、中心页豁免、自检脚本

