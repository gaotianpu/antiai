# Tag 受控词表

所有 YAML `tags` 字段必须从以下词表中选取，不加层级。每页 3-5 个 tag，至少覆盖「方法论 + 技术方法」两个维度。


## 文档原则：
- Agent 应先确认词表中没有语义接近的已有 Tag 再追加。新增 tag 应满足：①至少有 3 个页面会使用它，②不属于上表任何已有 tag 的同义词。追加后需同步更新本表。


## Tag 受控词表

| 分类 | 可用 Tag |
|:---|:---|
| 方法论 | `empirical-study`, `simulation`, `theoretical`, `practical-guide`, `survey`, `book` |
| 理论学科 | `information-theory`, `game-theory`, `cybernetics`, `systems-theory`, `evolutionary-biology`, `network-science` |
| 技术方法 | `HMM`, `NLP`, `topological-data-analysis`, `random-matrix-theory`, `signal-processing`, `monte-carlo`, `statistical-learning`, `robust-optimization`, `agent-based-model`, `machine-learning`, `overfitting`, `cross-validation`, `parameter-optimization`, `computer-vision`, `RL`, `autonomous-driving`, `imitation-learning`, `transformer`, `lane-detection` |
| 特殊 | `person`, `organization`, `index`, `log` |

> 2026-08-03 补录：`computer-vision`(184 页)、`RL`(23)、`autonomous-driving`(22)、`imitation-learning`(9)、`transformer`(5)、`lane-detection`(4)——均为实践已广泛使用且 ≥3 页的 tag，按维护规则补入。

**Tag 选择范式**：概念页推荐「方法论 + 1-2 个技术方法/学科」；实体页推荐「`person` 或 `organization` + 专长领域 + 方法论」；综合页推荐「核心主题 + `practical-guide`」。不要只用 `person` 一个 tag 标记实体页。




