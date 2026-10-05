"""
pdf2md_fix: PDF-to-Markdown 后处理修复工具

从 stdin 读入原始转换结果，按规则修复后输出到 stdout。
发现新的通用修复模式时追加到此文件。

用法: python pdf2md_fix.py < raw/XXXX.md > cleaned.md
"""

import re
import sys

LIGATURE_TABLE = str.maketrans({
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
})


def normalize_ligatures(lines):
    """将 PDF 提取常见的连字字符还原为普通字母组合"""
    return [line.translate(LIGATURE_TABLE) for line in lines]


def join_numbered_headings(lines):
    """合并「1.」单独成行 + 下一行标题的 PDF 提取模式"""
    pat = re.compile(r'^(\d+(?:\.\d+)*\.)\s*$')
    for i, line in enumerate(lines):
        m = pat.match(line.strip())
        if not m:
            continue
        j = i + 1
        while j < len(lines) and lines[j].strip() == '':
            j += 1
        if j >= len(lines):
            continue
        nxt = lines[j].strip()
        # 下一行需像标题：非空、不以小写开头、长度合理
        if nxt and not nxt[0].islower() and len(nxt) < 120:
            lines[i] = f"{m.group(1)} {nxt}"
            lines[j] = ''
    return lines


def fix_headings(lines):
    """章节号 → Markdown 标题层级；过滤正文中的编号列表项"""
    pat = re.compile(r'^#*\s*(\d+(?:\.\d+)*)\.\s+(.+)')
    for i, line in enumerate(lines):
        m = pat.match(line.strip())
        if not m:
            continue
        num, title = m.group(1), m.group(2)
        # 标题通常较短、不以标点结尾、不含正文数学/引用标记
        low = title.lower()
        if (len(title) > 50 or ',' in title or '. ' in title or ' e.g.' in low or ' i.e.' in low
                or title.endswith(('.', ',', ';', ':', '-', '=', '+', '−')) or '(cid:' in title):
            continue
        level = "#" * (min(num.count("."), 3) + 2)  # 1 → ##, 1.1 → ###, 1.1.1 → ####
        lines[i] = f"{level} {num}. {title}"
    return lines


def fix_roman_headings(lines):
    """罗马数字章节号 → Markdown 标题层级"""
    pat = re.compile(r'^(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s+(.+)$', re.IGNORECASE)
    for i, line in enumerate(lines):
        m = pat.match(line.strip())
        if not m:
            continue
        title = m.group(2)
        if len(title) > 50 or ',' in title or '. ' in title or '(cid:' in title:
            continue
        lines[i] = f"## {m.group(1).upper()}. {title}"
    return lines


def fix_common_headings(lines):
    """常见独立标题行 → Markdown 标题层级"""
    common = {
        'abstract': 'Abstract',
        'introduction': 'Introduction',
        'related work': 'Related Work',
        'method': 'Method',
        'methods': 'Methods',
        'experiments': 'Experiments',
        'experimental results': 'Experimental Results',
        'results': 'Results',
        'conclusion': 'Conclusion',
        'conclusions': 'Conclusions',
        'references': 'References',
    }
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if s in common:
            lines[i] = f"## {common[s]}"
    return lines


def fix_references(lines):
    """References 段落标题"""
    for i, line in enumerate(lines):
        if line.strip().lower() == "references":
            lines[i] = "## References"
            break
    return lines


def fix_abstract(lines):
    """Abstract 段落标题"""
    for i, line in enumerate(lines):
        if line.strip().lower() == "abstract":
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

    lines = normalize_ligatures(lines)
    lines = join_numbered_headings(lines)
    lines = fix_title(lines)
    lines = fix_headings(lines)
    lines = fix_roman_headings(lines)
    lines = fix_common_headings(lines)
    lines = fix_references(lines)
    lines = fix_abstract(lines)
    lines = fix_spurious_headings(lines)

    sys.stdout.write("\n".join(lines))


if __name__ == "__main__":
    main()
