"""
pdf2md_fix: PDF-to-Markdown 后处理修复工具

从 stdin 读入原始转换结果，按规则修复后输出到 stdout。
发现新的通用修复模式时追加到此文件。

用法: python pdf2md_fix.py < raw/XXXX.md > cleaned.md
"""

import re
import sys


def fix_headings(lines):
    """章节号 → Markdown 标题层级"""
    pat = re.compile(r'^(\d+(?:\.\d+)*)\.\s+(.+)')
    for i, line in enumerate(lines):
        m = pat.match(line.strip())
        if m:
            num, title = m.group(1), m.group(2)
            level = "#" * (min(num.count("."), 3) + 1)  # 1 → ##, 1.1 → ###, 1.1.1 → ####
            lines[i] = f"{level} {num}. {title}"
    return lines


def fix_references(lines):
    """References 段落标题"""
    for i, line in enumerate(lines):
        if line.strip() == "References":
            lines[i] = "## References"
            break
    return lines


def fix_abstract(lines):
    """Abstract 段落标题"""
    for i, line in enumerate(lines):
        if line.strip() == "Abstract":
            lines[i] = "## Abstract"
            break
    return lines


def fix_spurious_headings(lines, threshold=100):
    """修复引用条目被误识别为标题（编号 > threshold 的行去掉 ##）"""
    for i, line in enumerate(lines):
        m = re.match(r'^## (\d+)\.\s+', line.strip())
        if m and int(m.group(1)) > threshold:
            lines[i] = line[3:] if line.startswith("## ") else line
    return lines


def main():
    content = sys.stdin.read()
    lines = content.split("\n")

    lines = fix_headings(lines)
    lines = fix_references(lines)
    lines = fix_abstract(lines)
    lines = fix_spurious_headings(lines)

    sys.stdout.write("\n".join(lines))


if __name__ == "__main__":
    main()
