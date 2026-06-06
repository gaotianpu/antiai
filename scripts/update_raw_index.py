"""
从 wiki/sources/ 提取摘要和标签，同步更新 raw/index.md 的说明列。
每次新论文入库后执行：python3 scripts/update_raw_index.py
"""

import os, re, yaml

raw_info = {}
for fname in sorted(os.listdir('wiki/sources')):
    if fname == 'index.md' or not fname.endswith('.md'):
        continue
    with open(os.path.join('wiki/sources', fname)) as f:
        content = f.read()
    m = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not m:
        continue
    try:
        fm = yaml.safe_load(m.group(1))
    except:
        continue
    rm = re.search(r'\[阅读笔记\]\(\.\./\.\./raw/([^)]+)\)', content)
    if rm:
        raw_path = rm.group(1)
        tags = ', '.join(fm.get('tags', []))
        summary_m = re.search(r'- \*\*概述\*\*:\s*(.+)', content)
        summary = summary_m.group(1).strip() if summary_m else ''
        raw_info[raw_path] = (summary, tags)

with open('raw/index.md') as f:
    index = f.read()

def update_row(line):
    if not line.startswith('|') or '[[' not in line:
        return line
    m = re.search(r'\[\[([^\]]+)\]\]', line)
    if not m:
        return line
    fpath = m.group(1)
    if fpath in raw_info:
        summary, tags = raw_info[fpath]
        if summary:
            parts = line.split('|')
            if len(parts) >= 3:
                parts[2] = f" {summary}（`{tags}`）"
                return '|'.join(parts)
    return line

lines = index.split('\n')
new_lines = [update_row(l) for l in lines]
with open('raw/index.md', 'w') as f:
    f.write('\n'.join(new_lines))

updated = sum(1 for l in new_lines if l.startswith('|') and '（`' in l)
print(f"Updated {updated} entries")
