#!/usr/bin/env python3
"""
wiki-todos: 提取全部页面中的 TODO / [?] 标记 (Lint Step 2).

标记定义:
  TODO        — 大写 TODO 单词 (可带后续说明, 整行上下文)
  [?]         — 字面问号标记

用法:
  python .agents/skills/wiki-lint/wiki-todos.py              # 扫描全部 wiki
  python .agents/skills/wiki-lint/wiki-todos.py concepts     # 只扫 wiki/concepts
  python .agents/skills/wiki-lint/wiki-todos.py synthesis    # 只扫 wiki/synthesis

输出: 文件(相对 wiki/):行号: 上下文片段, 按文件路径排序.

说明: 本脚本只报告, 不写文件. 汇总到 backlog.md 或主页面由 Agent 处置.

退出码: 0 = 无标记, 1 = 存在标记, 2 = 参数错误
"""

import re
import sys
import pathlib

_PROJ_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
WIKI_DIR = _PROJ_ROOT / "wiki"

TODO_PATTERN = re.compile(r"\bTODO\b[^\n]*", re.IGNORECASE)
QUESTION_PATTERN = re.compile(r"\[\?\]")

CONTEXT_MAX = 80  # 输出上下文截断长度


def strip_code_blocks(body: str) -> list:
    """剥离 ``` 围栏与 `行内代码`: 其中的 TODO/[?] 是代码字面量, 不参与标记检测."""
    lines, in_code = [], False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            lines.append(line)
    return re.sub(r"`[^`\n]*`", "", "\n".join(lines)).splitlines()


def extract_todo_markers(line: str) -> list:
    """返回该行命中的标记类型列表, 如 ['TODO', '[?]']"""
    hits = []
    if TODO_PATTERN.search(line):
        hits.append("TODO")
    if QUESTION_PATTERN.search(line):
        hits.append("[?]")
    return hits


def clip(line: str, pos: int) -> str:
    """截取标记位置附近的上下文片段."""
    start = max(0, pos - 20)
    seg = line[start:pos + CONTEXT_MAX].strip()
    if start > 0:
        seg = "…" + seg
    return seg


def scan_dir(base: pathlib.Path) -> list:
    """扫描 base 下所有 .md, 返回 (file, lineno, marker, context) 列表."""
    findings = []
    for fp in sorted(base.rglob("*.md")):
        rel = fp.relative_to(WIKI_DIR)
        try:
            lines = strip_code_blocks(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for lineno, line in enumerate(lines, start=1):
            for marker in extract_todo_markers(line):
                pos = line.find("TODO") if marker == "TODO" else line.find("[?]")
                if pos < 0:
                    pos = 0
                findings.append((str(rel), lineno, marker, clip(line, pos)))
    return findings


def main():
    args = sys.argv[1:]
    target = args[0] if args and not args[0].startswith("--") else None

    if target:
        base = WIKI_DIR / target
        if not base.is_dir():
            print(f"[!] 子目录不存在: wiki/{target}")
            return 2
    else:
        base = WIKI_DIR

    findings = scan_dir(base)

    scope = str(base.relative_to(WIKI_DIR)) if base != WIKI_DIR else "全部 wiki"
    if not findings:
        print(f"✅ 无 TODO / [?] 标记 (扫描 {scope})")
        return 0

    n_todo = sum(1 for f in findings if f[2] == "TODO")
    n_q = sum(1 for f in findings if f[2] == "[?]")
    print(f"⚠️ 共 {len(findings)} 处标记 (TODO: {n_todo}, [?]: {n_q}):")
    for rel, lineno, marker, ctx in findings:
        print(f"  {rel}:{lineno}: [{marker}] {ctx}")
    print("\n处置: 完成项移除标记; 保留项汇总到 backlog.md 或主页面")
    return 1


if __name__ == "__main__":
    sys.exit(main())
