#!/usr/bin/env python3
"""
update_related_nodes.py — 不读文件全文，直接增删 related_nodes 列表条目

用法:
    python3 tools/update_related_nodes.py <文件> add <id> [<id2> ...]
    python3 tools/update_related_nodes.py <文件> remove <id> [<id2> ...]
    python3 tools/update_related_nodes.py <文件> list

示例:
    python3 tools/update_related_nodes.py wiki/synthesis/index.md add return_sources
    python3 tools/update_related_nodes.py wiki/concepts/index.md remove return_sources
    python3 tools/update_related_nodes.py wiki/synthesis/index.md list

适用场景:
    index.md 类文件（sources/concepts/synthesis/entities）的高频 related_nodes 维护。
    agent 场景：无需人肉读整个文件，直接把 id 塞进去即可。

限制:
    仅在 related_nodes 位于 frontmatter 内（第一组 --- 块）时有效。
    要求 related_nodes 列表写在单独一行（不跨行）。
"""

import re
import sys
from pathlib import Path


RN_PATTERN = re.compile(
    r'^(related_nodes:\s*\[)([^\]]*)(\])\s*$',
    re.MULTILINE
)

ITEM_PATTERN = re.compile(r"'([^']+)'|\"([^\"]+)\"|([^,\[\]\s]+)")


def parse_items(line: str) -> list[str]:
    """提取 related_nodes 列表条目, 兼容带引号 ('a', "a") 与无引号 (a) 两种写法."""
    m = re.search(r"\[([^\]]*)\]", line)
    body = m.group(1) if m else line
    items = []
    for m2 in ITEM_PATTERN.finditer(body):
        item = next((g for g in m2.groups() if g), "").strip().strip("'").strip('"')
        if item and item != "None":
            items.append(item)
    return items


def format_line(items: list[str]) -> str:
    quoted = [f"'{x}'" for x in items]
    return f"related_nodes: [{', '.join(quoted)}]"


def get_related_nodes_line(content: str) -> tuple[str, int, int] | None:
    """找到 frontmatter 中的 related_nodes 行，返回 (行文本, start, end)"""
    m = RN_PATTERN.search(content)
    if m:
        return m.group(0), m.start(), m.end()
    return None


def do_add(filepath: str, ids: list[str]) -> bool:
    path = Path(filepath)
    content = path.read_text(encoding='utf-8')

    match = get_related_nodes_line(content)
    if not match:
        print(f"  ✗ 未找到 related_nodes 行")
        return False

    line, start, end = match
    current = parse_items(line)
    existing = set(current)
    added = [x for x in ids if x not in existing]
    if not added:
        print(f"  ✓ 全部已存在，无需改动")
        return False

    new_items = sorted(set(current) | set(added))
    new_line = format_line(new_items)
    new_content = content[:start] + new_line + content[end:]

    path.write_text(new_content, encoding='utf-8')
    print(f"  ✓ 新增 {len(added)} 项: {', '.join(added)}")
    print(f"    {len(current)} → {len(new_items)} 项")
    return True


def do_remove(filepath: str, ids: list[str]) -> bool:
    path = Path(filepath)
    content = path.read_text(encoding='utf-8')

    match = get_related_nodes_line(content)
    if not match:
        print(f"  ✗ 未找到 related_nodes 行")
        return False

    line, start, end = match
    current = parse_items(line)
    removed = [x for x in ids if x in current]
    if not removed:
        print(f"  ✓ 未找到这些项，无需改动")
        return False

    new_items = sorted(set(current) - set(removed))
    new_line = format_line(new_items)
    new_content = content[:start] + new_line + content[end:]

    path.write_text(new_content, encoding='utf-8')
    print(f"  ✓ 移除 {len(removed)} 项: {', '.join(removed)}")
    print(f"    {len(current)} → {len(new_items)} 项")
    return True


def do_list(filepath: str) -> bool:
    content = Path(filepath).read_text(encoding='utf-8')
    match = get_related_nodes_line(content)
    if not match:
        print(f"  ✗ 未找到 related_nodes 行")
        return False

    line, _, _ = match
    items = parse_items(line)
    print(f"  共 {len(items)} 项:")
    for i, item in enumerate(items, 1):
        print(f"    {i:3d}. {item}")
    return True


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    filepath = sys.argv[1]
    action = sys.argv[2]
    ids = sys.argv[3:]

    if not Path(filepath).exists():
        print(f"  ✗ 文件不存在: {filepath}")
        sys.exit(1)

    if action == 'add':
        if not ids:
            print("  ✗ add 操作需要至少一个 id")
            sys.exit(1)
        do_add(filepath, ids)
    elif action == 'remove':
        if not ids:
            print("  ✗ remove 操作需要至少一个 id")
            sys.exit(1)
        do_remove(filepath, ids)
    elif action == 'list':
        do_list(filepath)
    else:
        print(f"  ✗ 未知操作: {action}，可用 add / remove / list")
        sys.exit(1)


if __name__ == '__main__':
    main()
