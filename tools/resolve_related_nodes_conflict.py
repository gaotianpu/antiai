#!/usr/bin/env python3
"""
resolve_related_nodes_conflict.py — 解决 git pull 后 related_nodes 列表冲突

用法:
    python3 tools/resolve_related_nodes_conflict.py <冲突文件路径>

功能:
    1. 扫描文件中 git 冲突标记 (<<<<<<< / ======= / >>>>>>>)
    2. 提取两个版本的 related_nodes: [...] 列表
    3. 取并集去重，按字母序重排
    4. 替换冲突区域为合并后的内容

适用场景:
    wiki/{sources,concepts,synthesis}/index.md 等高频 related_nodes 冲突

原理:
    只处理 related_nodes 字段的冲突，其他冲突区域会跳过并给出警告。
"""

import re
import sys
from pathlib import Path


CONFLICT_PATTERN = re.compile(
    r'<<<<<<< .*?\n'
    r'(related_nodes: \[.*?\])\n'
    r'=======\n'
    r'(related_nodes: \[.*?\])\n'
    r'>>>>>>> [^\n]*',
    re.DOTALL
)

ITEM_PATTERN = re.compile(r"'([^']+)'|\"([^\"]+)\"|([^,\[\]\s]+)")


def parse_list(related_nodes_line: str) -> list[str]:
    """从 related_nodes 行提取列表, 兼容带引号与无引号两种写法."""
    m = re.search(r"\[([^\]]*)\]", related_nodes_line)
    body = m.group(1) if m else related_nodes_line
    items = []
    for m2 in ITEM_PATTERN.finditer(body):
        item = next((g for g in m2.groups() if g), "").strip().strip("'").strip('"')
        if item and item != "None":
            items.append(item)
    return items


def format_list(items: list[str]) -> str:
    """将列表格式化为 related_nodes: ['a', 'b', 'c']"""
    quoted = [f"'{x}'" for x in items]
    return "related_nodes: [" + ", ".join(quoted) + "]"


def resolve_conflict(filepath: str) -> bool:
    """解析文件中的 related_nodes 冲突，有改动返回 True"""
    path = Path(filepath)
    content = path.read_text(encoding='utf-8')

    # 检查是否有冲突标记
    if '<<<<<<<' not in content:
        print(f"  ✗ 无冲突标记，跳过")
        return False

    # 查找 related_nodes 冲突
    matches = list(CONFLICT_PATTERN.finditer(content))
    if not matches:
        print(f"  ⚠ 有冲突标记但未匹配到 related_nodes 格式（非列表冲突），请手动处理")
        return False

    new_content = content
    resolved_count = 0

    for match in reversed(matches):  # 从后往前替换，避免偏移
        upstream_line = match.group(1)
        stashed_line = match.group(2)

        upstream_items = parse_list(upstream_line)
        stashed_items = parse_list(stashed_line)

        # 并集去重 + 字母序
        merged = sorted(set(upstream_items) | set(stashed_items))
        merged_line = format_list(merged)

        # 计算差异
        only_upstream = set(upstream_items) - set(stashed_items)
        only_stashed = set(stashed_items) - set(upstream_items)

        print(f"  ✓ 合并完成: {len(upstream_items)} + {len(stashed_items)} → {len(merged)} 项")
        if only_upstream:
            print(f"    +remote: {', '.join(sorted(only_upstream))}")
        if only_stashed:
            print(f"    +local:  {', '.join(sorted(only_stashed))}")

        new_content = new_content[:match.start()] + merged_line + new_content[match.end():]
        resolved_count += 1

    path.write_text(new_content, encoding='utf-8')
    print(f"  ▶ 已保存 {resolved_count} 处冲突到 {filepath}")
    return True


def main():
    if len(sys.argv) < 2:
        print("用法: python3 tools/resolve_related_nodes_conflict.py <文件路径1> [文件路径2 ...]")
        print("示例: python3 tools/resolve_related_nodes_conflict.py wiki/sources/index.md wiki/synthesis/index.md")
        sys.exit(1)

    total = 0
    for filepath in sys.argv[1:]:
        if not Path(filepath).exists():
            print(f"  ✗ 文件不存在: {filepath}")
            continue
        print(f"\n📄 {filepath}")
        if resolve_conflict(filepath):
            total += 1

    print(f"\n{'='*40}")
    print(f"处理完成: {total} 个文件已解决冲突")
    print(f"提示: 解决后执行 git add <文件> 标记已解决")


if __name__ == '__main__':
    main()
