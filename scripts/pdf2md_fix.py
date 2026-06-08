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


def fix_title(lines):
    """将文件首行的论文标题补上 #"""
    # 如果第一行已经是标题，跳过
    for line in lines:
        if line.strip():
            if line.startswith('#'):
                return lines
            break

    # 找到标题块的结束位置（第一个非标题行）
    author_pat = re.compile(r'^[A-Z][A-Za-z\-\'*0-9]+ [A-Za-z\-\'*0-9]+[，,]')
    title_end = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if s in ('Abstract', '## Abstract', 'DeepSeek-AI'):
            title_end = i
            break
        if '@' in s:
            title_end = i
            break
        if author_pat.match(s):
            title_end = i
            break

    if title_end <= 0:
        # 回退：把第一个非空行当标题
        title_end = 0
        for i, line in enumerate(lines):
            if line.strip():
                title_end = i
                break
        # 只有一行就只改那一行
        if title_end >= 0:
            lines[title_end] = f"# {lines[title_end].strip()}"
        return lines

    # 收集标题块（title_end 前的非空行）
    title_parts = []
    first_idx = -1
    for i in range(title_end):
        s = lines[i].strip()
        if s:
            if first_idx < 0:
                first_idx = i
            title_parts.append(s)

    if title_parts:
        title = ' '.join(title_parts)
        lines[first_idx] = f"# {title}"
        for i in range(first_idx + 1, title_end):
            lines[i] = ''
    return lines


def main():
    content = sys.stdin.read()
    lines = content.split("\n")

    lines = fix_title(lines)
    lines = fix_headings(lines)
    lines = fix_references(lines)
    lines = fix_abstract(lines)
    lines = fix_spurious_headings(lines)

    sys.stdout.write("\n".join(lines))


if __name__ == "__main__":
    main()
