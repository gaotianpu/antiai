"""
检查 wiki/{sources,concepts,entities}/index.md 是否收录了目录下所有文件。
定期执行: python3 scripts/check_index_completeness.py
"""

import os, re

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
checks = {
    'wiki/sources': 'wiki/sources/index.md',
    'wiki/concepts': 'wiki/concepts/index.md',
    'wiki/entities': 'wiki/entities/index.md',
}

all_ok = True
for dir_path, index_path in checks.items():
    full_dir = os.path.join(PROJECT, dir_path)
    full_index = os.path.join(PROJECT, index_path)

    files = set(f.replace('.md', '') for f in os.listdir(full_dir)
                if f.endswith('.md') and f != 'index.md')

    with open(full_index) as f:
        indexed = set(re.findall(r'\[\[([^\]]+)\]\]', f.read()))

    missing = files - indexed
    extra = indexed - files

    print(f'{dir_path}/: {len(files)} files, {len(indexed)} indexed')
    if missing:
        print(f'  ❌ 未索引: {", ".join(sorted(missing))}')
        all_ok = False
    if extra:
        print(f'  ⚠️ 不存在却引用的: {", ".join(sorted(extra))}')
        all_ok = False

if all_ok:
    print('\n全部索引完整 ✅')
else:
    print('\n存在缺失项，请补充 ❌')
