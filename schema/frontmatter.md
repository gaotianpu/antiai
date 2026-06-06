# YAML Frontmatter 字段规范

所有新建页面必须包含以下 YAML frontmatter：

```yaml
---
id: [唯一标识，与文件名一致，不含路径/扩展名]
type: [source | concept | entity | synthesis | changelog]
tags: [3-5个分类标签，平铺无层级]
aliases: [中文名，英文别名，缩写，常见拼写变体]
related_nodes: [双向链接的页面 id 列表]
---
```

## 字段说明

| 字段 | 必填 | 说明 |
|:---|:---:|:---|
| `id` | ✅ | 与文件名一致（不含 `.md` 和路径）。用于 Dataview 查询和跨页引用 |
| `type` | ✅ | 页面类型（source / concept / entity / synthesis / changelog）。速查见 `AGENTS.md` §wiki/ 页面类型速查 |
| `tags` | ✅ | 3-5 个 flat tag，从 `schema/tags.md` 受控词表中选择 |
| `aliases` | ✅ | Obsidian 原生识别此字段实现多名称搜索。至少包含中英文两个别名 |
| `related_nodes` | ✅ | 本页引用的其他 page id 列表，用于 Graph 展示和双向链接审计 |
| `last_verified` | | 可选。YYYY-MM-DD 格式，标记内容最后核实日期。Source / Concept 类页面推荐填写 |
