---
name: wiki-lint
description: 定期或按需对 wiki 执行质量检查：孤岛扫描、死链清理、冲突检测、TODOS 提取、表格转义检查、related_nodes 对称性验证、索引完整性检查。
metadata:
  variables:
    - WIKI_DIR: Wiki 页面目录（默认 $PROJECT_ROOT/wiki）
    - SCHEMA_DIR: 规范文件目录（默认 $PROJECT_ROOT/schema）
    - PROJECT_ROOT: 项目根目录（默认当前工作目录）
---

# Wiki Lint — 质量校验

## 配置变量

| 变量 | 默认值 | 说明 |
|:---|:---|:---|
| `PROJECT_ROOT` | `$PWD` | 项目根目录 |
| `WIKI_DIR` | `$PROJECT_ROOT/wiki` | wiki 页面目录 |
| `SCHEMA_DIR` | `$PROJECT_ROOT/schema` | 规范文件目录 |

```bash
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
WIKI_DIR="${WIKI_DIR:-$PROJECT_ROOT/wiki}"
SCHEMA_DIR="${SCHEMA_DIR:-$PROJECT_ROOT/schema}"
```

---

Agent 需定期（或在 [[wiki-query]] 检索质量下降时）自发执行 Lint 动作：

## 七项检查

### 1. 孤岛扫描

寻找没有任何入链的 `.md` 文件。

```bash
# 找出所有 wiki 文件
all_files=$(find "$WIKI_DIR" -name '*.md' ! -name 'index.md' ! -path '*/changelog/*')

# 对每个文件，检查是否有其他文件通过 [[id]] 引用它
for f in $all_files; do
  id=$(basename "$f" .md)
  # 跳过 index.md 主入口
  [ "$id" = "index" ] && continue
  # 检查是否有其他文件链接到此 id
  refs=$(grep -rl "\[\[$id\]\]" "$WIKI_DIR" --include='*.md' 2>/dev/null | grep -v "$f" | head -1)
  if [ -z "$refs" ]; then
    echo "❌ 孤岛: $f"
  fi
done
```

排除：`index.md`、`changelog/` 下的日志文件。

### 2. 死链清理

检查 `[[id]]` 或 `[[path]]` 指向的文件是否存在。**注意：GNU grep BRE 的 `[^\\]\\[]` 字符类会静默失效，不要用 grep 提取链接，一律用下方 Python 脚本。**

```bash
python3 -c "
import os, re
from collections import defaultdict

WIKI = '$WIKI_DIR'
# 1. 收集所有页面名 + frontmatter aliases（别名也是有效链接目标）
valid = set()
for root, dirs, files in os.walk(WIKI):
    for f in files:
        if f.endswith('.md'):
            valid.add(f[:-3])
            head = open(os.path.join(root, f)).read()[:2000]
            m = re.search(r'aliases:\\s*\\[([^\\]]+)\\]', head)
            if m:
                for a in m.group(1).split(','):
                    valid.add(a.strip().strip('\"\''))

# 2. 提取链接：先剥离表格转义 \\|，再依次剥离 |alias 与 #anchor
dead = defaultdict(list)
pat = re.compile(r'\\[\\[([^\\]\\[]*)\\]\\]')
for root, dirs, files in os.walk(WIKI):
    for f in files:
        if not f.endswith('.md') or root.endswith('changelog') or f == 'todo.md':
            continue
        path = os.path.join(root, f)
        for m in pat.finditer(open(path).read()):
            raw = m.group(1).replace('\\\\|', '\x00')
            target = raw.split('\x00')[0].split('|')[0].strip().split('#')[0].strip()
            if '://' in target or '/' in target:
                continue
            if target not in valid:
                dead[target].append(os.path.relpath(path, WIKI))

if dead:
    for t, srcs in sorted(dead.items()):
        print(f'❌ [[{t}]] ← {srcs}')
else:
    print('✅ 无死链')
"
```

**死链二分类**：命中后先判性质再处置——
- **概念缺失**：目标像领域概念名（如 `[[bert]]`、`[[focal_loss]]`），且无同名/别名页面 → 是 Source 页先入库、概念页未补建，登记到 `wiki/todo.md` 或直接按 wiki-ingest 补建，**不是笔误**
- **笔误死链**：目标明显拼写错误或指向已删页面 → 修正链接指向

```bash
# raw/index.md 中的链接（指向 raw/ 文件）
python3 -c "
import os, re
index = open('$PROJECT_ROOT/raw/index.md').read()
pat = re.compile(r'\\[\\[([^\\]\\[]*)\\]\\]')
for m in pat.finditer(index):
    raw = m.group(1).replace('\\\\|', '\x00')
    target = raw.split('\x00')[0].split('|')[0].strip()
    if '://' in target:
        continue
    if not os.path.isfile(os.path.join('$PROJECT_ROOT/raw', target)):
        print(f'❌ raw 死链: [[{target}]]')
"
```

### 3. 冲突检测

检查 `$WIKI_DIR/concepts/` 下是否存在对同一术语的不同定义。

```bash
python3 -c "
import os, re
from collections import defaultdict

terms = defaultdict(list)
for f in os.listdir('$WIKI_DIR/concepts'):
    if not f.endswith('.md') or f == 'index.md':
        continue
    with open(os.path.join('$WIKI_DIR/concepts', f)) as fh:
        content = fh.read()
    aliases = re.findall(r'aliases:\s*\[([^\]]+)\]', content)
    if aliases:
        for a in aliases[0].split(','):
            a = a.strip().strip('\"\'')
            terms[a].append(f.replace('.md',''))

for term, pages in terms.items():
    if len(pages) > 1:
        print(f'⚠️  术语「{term}」在多个概念页中出现: {pages}')
"
```

如有 `$SCHEMA_DIR/concept_dedup.md`，参照其规则合并。

### 4. TODOS 提取

扫描所有页面中的 `TODO` 或 `[?]` 标记，汇总至主页面。

```bash
echo '# TODO 汇总' > "$WIKI_DIR/todo.md"
grep -rn 'TODO\|\[?\]' "$WIKI_DIR" --include='*.md' \
  --exclude='todo.md' --exclude-dir='changelog' \
  | sed 's/^/ - /' >> "$WIKI_DIR/todo.md"
echo "✅ TODOs 已汇总到 wiki/todo.md（$(grep -c '^-' "$WIKI_DIR/todo.md") 条）"
```

### 5. 表格转义检查

检查表格中的 `[[page|alias]]` 是否已正确转义为 `[[page\|alias]]`。

```bash
# 找出表格行中未转义的竖线（-E 正则避免 [[page]] 无别名的贪婪误报）
grep -rnE '^\|.*\[\[[^]]*\|[^]]*\]\]' "$WIKI_DIR" --include='*.md' \
  | grep -v '\\|' \
  | head -20
```

命中的行需要将 `[[page|alias]]` 改为 `[[page\|alias]]`。参见 `AGENTS.md §2`。

### 6. related_nodes 对称性

检查新建 page 的双向引用是否完整。**注意方向性豁免**：

> 按 `schema/best_practices.md` §12，related_nodes 应有方向性，中心页（如 attention_mechanism 被几十页引用）**不需要**反向平铺所有引用者。只检查**最近入库/新建的页面**，历史遗留的缺失不批量修改。

```bash
# 检查 last_verified 距今较近的页面
find "$WIKI_DIR" -name '*.md' -newer "$WIKI_DIR/index.md" | while read f; do
  id=$(basename "$f" .md)
  related=$(grep 'related_nodes:' "$f" | grep -oP '\[([^\]]+)\]' | tr -d '[]", ')
  for target in $related; do
    target_file=$(find "$WIKI_DIR" -name "${target}.md" 2>/dev/null | head -1)
    [ -z "$target_file" ] && echo "❌ $id → [[$target]] 目标不存在" && continue
    grep -q "\[\[$id\]\]" "$target_file" || echo "⚠️  $id → [[$target]] 缺少反向引用"
  done
done
```

### 7. 索引完整性检查

```bash
python3 "$PROJECT_ROOT/scripts/check_index_completeness.py"
```

验证 `wiki/{sources,concepts,entities}/index.md` 及 `raw/index.md` 是否收录了所有文件。定期执行或大批量摄入后执行。
