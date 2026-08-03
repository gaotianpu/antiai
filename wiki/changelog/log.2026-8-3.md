---
type: log
---
## [Lint] wiki-lint 全量质量检查

### 修复

- **表格转义** ×3: `wiki/synthesis/deepseek_papers.md` 中 `[[mixture_of_experts|MoE]]`、`[[reinforcement_learning|RL]]`、`[[chain_of_thought|CoT]]` → `\|` 转义
- **raw 索引补录** ×7: raw/index.md 新增 2024/2025 分组（6 篇 DeepSeek 论文 arXiv 原文），2014 分组 `[[cnn/googlenet.md]]` → `[[cnn/inception_v1.md]]`（文件实际存在名）
- **死链修复** ×2: `wiki/sources/szegedy_2014_googlenet.md` 阅读笔记链接 `googlenet.md` → `inception_v1.md`；删除空占位文件 `raw/cnn/googlenet.md`（0 字节）
- **别名冲突清理** ×9: 领域页/上位页 aliases 移除越界子概念名——information_theory（熵/KL散度/互信息）、optimization_fundamentals（凸优化）、calculus（链式法则）、linear_algebra（矩阵运算）、optimizer（梯度下降/SGD）、attention_mechanism（self-attention/自注意力）
- **related_nodes 双向补全** ×4: mixture_of_experts（+bi_2024_deepseek_llm, deepseek_papers）、conditional_memory（+deepseek_papers）、multi_head_latent_attention（+wei_2025_deepseek_ocr）
- **脚本改进**: `scripts/check_index_completeness.py` 排除 10 个非论文文件（7 个子目录 README + NewBeer/openai/deepseek_index 导航笔记）

### 生成

- **wiki/todo.md**: 145 种缺失概念页待办（Source 页引用但 concepts/ 未创建，156 次引用）

### 检查通过项

- 孤岛扫描: 0 孤岛
- wiki 内部死链: 0（除上述已记录的概念缺失待办）
- raw/index.md 死链: 0
- 索引完整性: wiki 3 目录 + raw 全部通过
- TODO 标记: 无（除新建 todo.md）
