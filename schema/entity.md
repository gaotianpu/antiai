# 实体识别与维护规范 (Entity: Person & Organization)

> **适用范围**：`wiki/entities/` 目录下的人物和组织实体。
> **不适用范围**：概念、指标、工具、方法、数据——这些归属 `wiki/concepts/`，见 `schema/concept.md`。

---

## 1. 实体定义

**实体 (Entity)**：具有独立身份、可被唯一指向的**单数客观存在**。本 Wiki 中实体仅指：

| 类型 | 示例 | 判断标准 |
|:---|:---|:---|
| **人物** | `zura_kakushadze`, `claude_shannon` | "这个人客观存在，无论是否有论文研究他" |
| **组织** | `worldquant_llc` | "这家公司客观存在，可以指向" |

> **与 Concept 的分界线**："这是谁/哪家公司"（Entity），vs "这是什么"（Concept）。
> 
> BAD: 把 `sharpe_ratio.md` 放在 `entities/` — 它是一个抽象公式，属于概念。
> GOOD: 把 `claude_shannon.md` 放在 `entities/` — 他是一个具体的人。

---

## 2. 文件命名规范

- **格式**: 蛇形命名法 (snake_case)，全小写
- **人物**: 名_姓 或 常用缩写
  - `claude_shannon.md`, `marcos_lopez_de_prado.md`, `zura_kakushadze.md`
- **组织**: 公司名缩写或全称
  - `worldquant_llc.md`
- 避免特殊字符和中文

---

## 3. 实体页面结构

```markdown
---
id: [与文件名一致]
type: entity
tags: [person 或 organization, 相关领域tag]
aliases: [中文名, 英文全名, 缩写]
related_nodes: [相关的概念/源/实体 page id]
---

# 姓名 (English Name)（中文名）

## 概述
[1-2 句话：身份 / 核心贡献 / 与 Wiki 的关联]

## 关键贡献
- **[[Source_ID]]**（年份）：[一句话贡献描述]
- ...

## 学术关联
- 任职/归属机构
- 研究方向
- 合作者（链接到其他实体页）

## 引用资料
1. [[Source_ID]] — [此人在该资料中的角色]
```

---

## 4. 创建/更新流程

### 4.1 新建实体
1. 检查 `wiki/entities/` 是否已存在
2. **别名碰撞检测**：搜索所有已有实体页的 `aliases` 字段，确认无同人/同组织重复：

   ```bash
   # 人物：搜索全名和缩写
   grep -ril "Yoshua Bengio\|Y. Bengio\|bengio" wiki/entities/ 2>/dev/null
   # 组织：搜索常用变体
   grep -ril "DeepMind\|Google DeepMind\|deepmind" wiki/entities/ 2>/dev/null
   ```

3. 按 §3 模板创建
4. 在 `wiki/entities/index.md` 追加条目
5. 确保被引用的 source/concept 页的 `related_nodes` 包含此实体

> **去重原则**：同一人物/组织只建一个实体页。发现重复时，选择信息更全的页面保留，另一个重定向或合并。

### 4.2 关联维护
- 当创建新 source 页涉及某人物时，检查该人物实体页是否需要补充新增贡献
- 人物与人物之间的合作关系通过 `related_nodes` 双向链接

---

## 5. 索引管理

`wiki/entities/index.md` 按领域分组：

```markdown
# 实体索引

## 按类型

### 人物
**深度学习** → **强化学习** → **NLP** → **计算机视觉** → ...

### 组织
- [[deepseek_ai]] — AI 研究实验室
- [[google]] — 科技公司
- [[microsoft]] — 科技公司
- [[openai]] — AI 研究机构
```

---

## 6. 质量标准

- **准确性**: 忠实于原始资料
- **精简**: 实体页是"名片页"，不展开长篇论述。详细内容留给 source 或 concept 页
- **贡献排序**: 「关键贡献」按年份升序排列（而非重要性或添加顺序）
- **互连性**: 通过 `related_nodes` + `[[双向链接]]` 建立到 source/concept 的引用
- **中英双语**: `aliases` 至少包含中文名和英文名
