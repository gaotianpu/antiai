---
type: log
---
## [Update] kaiming_he 实体页补录 2023–2026 年工作

- 更新 Entity: [[kaiming_he]] — 关键贡献列表追加 25 篇（2023-12 至 2026-07，arXiv 直链，尚未建 source 页）
- 更正: aliases 去重（`何恺明` 重复 ×2 → ×1）；概述补充 2024 年起任职 MIT、研究重心转向生成模型
- 更新: `last_verified` → 2026-09-30
- 依据: arXiv API 全量核查 `au:"Kaiming He"`（2014–2026 共 80 篇）；量子调度论文（2405.16380）经论文邮箱 kaiming@mit.edu 核实为本人
- 说明: 按方案 C 执行——仅更新实体页，不创建新 source/concept 页

## [Update] deepseek_ai 实体页补录至 2026 年

- 更新 Entity: [[deepseek_ai]] — 关键贡献列表重排为时间序并扩至 29 项（2024-01 至 2026-09）：9 项已有 source 页 + 19 项 arXiv/官网直链 + 1 项无链接技术报告
- 新增收录: DeepSeekMoE、DeepSeek-VL、Prover 系列、Coder-V2、Fire-Flyer AI-HPC、Janus 系列、VL2、NSA、Insights into DeepSeek-V3、Math-V2、Thinking with Visual Primitives、mHC、OCR 2、V4、V4.1-Flash、DSec
- 修复: [[shao_2024_deepseekmath]]、[[deepseek_2025_v32]] 的 related_nodes 补入 deepseek_ai（双向链接对称性）；实体页反链同步补入
- 更新: 概述代表作 V2/V3/R1 → V3/R1/V4；`last_verified` → 2026-09-30
- 说明: 按方案 C 执行——仅更新实体页，不创建新 source/concept 页

## [Docs] 保存 changelog 机制讨论至 best_practices

- 更新 Schema: `schema/best_practices.md` 新增 §14「changelog 与 git log 的分工」——记录约束来源（AGENTS.md 表行 + wiki-ingest 阶段 4 + parking-lot）、五维分工对比、决议（维持双写）、已知 type 不一致（维持 `type: log` 现状）

## [Update] 目标检测 IoU 损失系列 source 页 + 概念页 + 综述页

- 新建 Source ×8: [[everingham_2010_pascal_voc]]、[[yu_2016_unitbox]]、[[rezatofighi_2019_giou]]、[[zheng_2020_diou]]、[[zhang_2022_eiou]]、[[gevorgyan_2022_siou]]、[[tong_2023_wiou]]、[[he_2021_alpha_iou]]
- 新建 Concept: [[iou_loss]] — IoU 及其改进变体总表（GIoU/DIoU/CIoU/EIoU/SIoU/WIoU/Alpha-IoU/Focal-EIoU）
- 新建 Synthesis: [[iou_loss_survey]] — 目标检测 IoU 损失演进综述；创建前已检查全库无同题综述页
- 更新 Index: `wiki/sources/index.md`（2010/2016/2019/2020/2021/2022/2023）、`wiki/concepts/index.md`、`wiki/synthesis/index.md`、`wiki/index.md`
- 双向链接: [[loss_function]]、[[object_detection]] 补充 `related_nodes` 与正文链接
- 数据来源: arXiv API/Crossref 核对标题、作者、DOI；arXiv 原文 PDF 核对 UnitBox/EIoU/SIoU/WIoU/Alpha-IoU 公式
- 说明: DIoU/CIoU 共用 [[zheng_2020_diou]]，EIoU/Focal-EIoU 共用 [[zhang_2022_eiou]]；IoU 度量的评测出处单列为 [[everingham_2010_pascal_voc]]
