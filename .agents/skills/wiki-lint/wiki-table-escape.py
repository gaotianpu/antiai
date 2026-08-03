#!/usr/bin/env python3
"""
wiki-table-escape: 检查 [[page|alias]] 表格转义合规性 (Lint Step 3).

规则 (见 AGENTS.md 编辑器协议):
  表格行中  [[page|alias]] 必须转义为 [[page\\|alias]] (反斜杠 + 竖线)
  非表格段中 [[page\\|alias]] 不应出现 (反斜杠多余)

边界:
  数学公式中的 \\| (如 $\\|d\\|$) 不在 [[ ]] 内, 一律不碰.
  表格行判定: 以 | 开头 (markdown 表格允许行尾省略 |).

用法:
  python .agents/skills/wiki-lint/wiki-table-escape.py          # 只报告
  python .agents/skills/wiki-lint/wiki-table-escape.py --fix    # 报告并自动修复
  python .agents/skills/wiki-lint/wiki-table-escape.py synthesis # 只扫指定子目录

退出码: 0 = 全部合规, 1 = 存在问题, 2 = 参数错误
"""

import re
import sys
import pathlib

_PROJ_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
WIKI_DIR = _PROJ_ROOT / "wiki"

# 表格行: 以 | 开头 (允许缩进)
TABLE_ROW = re.compile(r"^\s*\|")

# 未转义: [[page|alias]] — | 前不能是 \ (负向后顾避免误吃 [[page\|alias]])
LINK_RAW = re.compile(r"\[\[([^\[\]\n]+?)(?<!\\)\|([^\[\]\n]+?)\]\]")
# 已转义: [[page\|alias]]
LINK_ESCAPED = re.compile(r"\[\[([^\[\]\n]+?)\\\|([^\[\]\n]+?)\]\]")


def strip_code_blocks(body: str) -> list:
    """剥离 ``` 围栏与 `行内代码`: 其中的 | 表格样式与 [[...]] 是代码字面量, 不参与转义检测."""
    lines, in_code = [], False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            lines.append(line)
    return re.sub(r"`[^`\n]*`", "", "\n".join(lines)).splitlines()


def scan_line(line: str, in_table: bool) -> list:
    """返回该行问题列表: [(kind, matched_text, fixed_text), ...]
    kind: 'raw-in-table' 表格内未转义 / 'escaped-outside' 非表格误转义
    """
    issues = []
    if in_table:
        for m in LINK_RAW.finditer(line):
            issues.append(("raw-in-table", m.group(0),
                           f"[[{m.group(1)}\\|{m.group(2)}]]"))
    else:
        for m in LINK_ESCAPED.finditer(line):
            issues.append(("escaped-outside", m.group(0),
                           f"[[{m.group(1)}|{m.group(2)}]]"))
    return issues


def fix_line(line: str, in_table: bool) -> str:
    """按规则修复一行 (仅 wikilink 转义, 不动数学公式)."""
    if in_table:
        line = LINK_RAW.sub(lambda m: f"[[{m.group(1)}\\|{m.group(2)}]]", line)
    else:
        line = LINK_ESCAPED.sub(lambda m: f"[[{m.group(1)}|{m.group(2)}]]", line)
    return line


def scan_dir(base: pathlib.Path, fix: bool) -> list:
    """返回 (rel, lineno, kind, original, fixed) 列表; fix=True 时就地写回."""
    findings = []
    for fp in sorted(base.rglob("*.md")):
        rel = fp.relative_to(WIKI_DIR)
        try:
            lines = strip_code_blocks(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        changed = False
        for lineno, line in enumerate(lines, start=1):
            in_table = bool(TABLE_ROW.match(line))
            issues = scan_line(line, in_table)
            if issues:
                for kind, orig, fixed in issues:
                    findings.append((str(rel), lineno, kind, orig, fixed))
                if fix:
                    lines[lineno - 1] = fix_line(line, in_table)
                    changed = True
        if changed:
            fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return findings


def main():
    args = sys.argv[1:]
    fix = "--fix" in args
    target = next((a for a in args if not a.startswith("--")), None)

    if target:
        base = WIKI_DIR / target
        if not base.is_dir():
            print(f"[!] 子目录不存在: wiki/{target}")
            return 2
    else:
        base = WIKI_DIR

    findings = scan_dir(base, fix)

    if not findings:
        print(f"✅ 表格转义全部合规 (扫描 {base.relative_to(WIKI_DIR)})")
        return 0

    for kind, label in (("raw-in-table", "表格内未转义 (应加 \\ )"),
                        ("escaped-outside", "非表格误转义 (应去 \\ )")):
        rows = [f for f in findings if f[2] == kind]
        if not rows:
            continue
        print(f"⚠️ {label} {len(rows)} 处:")
        for rel, lineno, _, orig, fixed in rows:
            print(f"  {rel}:{lineno}: {orig}")
            if fix:
                print(f"    → {fixed} (已修复)")
        print()

    if not fix:
        print("提示: 加 --fix 可自动修复以上全部转义问题")
    return 1


if __name__ == "__main__":
    sys.exit(main())
