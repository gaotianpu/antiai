#!/usr/bin/env python3
"""
wiki-concept-conflicts: 概念冲突候选检测 (Lint Step 1, 半自动).

原理: 脚本只做机械部分 — 从 wiki/concepts/ 各页的 id / aliases / H1 标题中
     找名称重叠的候选对; 是否真为"同一术语的不同定义"仍需人工确认
     (见 schema/concept_dedup.md).

候选规则:
  1. 别名/标题/id 直接重叠  — A 与 B 的 token 集合有交集 (如共同中文别名)
  2. 名称包含               — A 的 id 词集是 B 的 id 词集的子集 (如 alpha vs alpha_delay)
  3. 标题主名重叠           — H1 括号前主名相同或互为别名

用法:
  python .agents/skills/wiki-lint/wiki-concept-conflicts.py
  python .agents/skills/wiki-lint/wiki-concept-conflicts.py --all  # 包含 entities/factors

输出: 候选对 + 命中原因, 按重叠强度排序. 仅提示, 不自动修改.

退出码: 0 = 无候选, 1 = 存在候选, 2 = 参数错误
"""

import re
import sys
import pathlib
import yaml

_PROJ_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
WIKI_DIR = _PROJ_ROOT / "wiki"

SCOPE = {
    "concepts": (WIKI_DIR / "concepts", "概念"),
    "entities": (WIKI_DIR / "entities", "实体"),
}

H1_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)

# aliases 条目: 兼容带引号 ('a', "a") 与无引号 (a) 写法.
# 第三组允许内部空格 (如 Alpha #1 整体是一个条目, 不能拆成 Alpha/#1 碎片)
ITEM_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"|([^,\[\]]+)")


def strip_code_blocks(body: str) -> str:
    """剥离 ``` 围栏与 `行内代码`: 其中的 # 行是代码注释, 不能当 H1 标题."""
    lines, in_code = [], False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            lines.append(line)
    return re.sub(r"`[^`\n]*`", "", "\n".join(lines))


def extract_flow_list(fm_text: str, key: str) -> list:
    """从 frontmatter 原文正则提取单行 flow 列表字段 (aliases 等), YAML 失败时的兜底."""
    m = re.search(rf"^{key}:\s*\[([^\]]*)\]", fm_text, re.MULTILINE)
    if not m:
        return []
    items = []
    for mm in ITEM_RE.finditer(m.group(1)):
        item = next((g for g in mm.groups() if g), "").strip().strip("'").strip('"')
        if item:
            items.append(item)
    return items


def parse_page(fp: pathlib.Path) -> dict | None:
    """返回 {id, tokens, main_title, file}. tokens = 归一化的 id+aliases+标题主名.

    id 用正则从 frontmatter 原文提取 (不依赖 YAML):
      纯数字 id 会被 YAML 浮点化; aliases 含 ' #xx' 注释会导致 ParserError 整页失败
      (如 factors 下 102 个 alpha101#N 页的 aliases: [..., Alpha #1, ...]).
    """
    try:
        content = fp.read_text(encoding="utf-8")
    except Exception:
        return None
    m = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not m:
        return None
    fm_text = m.group(1)

    # id: 正则优先
    lm = re.search(r"^id:\s*(.*)$", fm_text, re.MULTILINE)
    if not lm:
        return None
    pid = lm.group(1).strip().strip("'\"")
    if not pid:
        return None

    # 跳过 index / log 页 (目录页与审计日志不参与冲突检测)
    ptype = ""
    try:
        fm = yaml.safe_load(fm_text)
        if isinstance(fm, dict):
            ptype = str(fm.get("type", "")).strip()
    except Exception:
        pass
    if ptype in ("index", "log"):
        return None

    # H1 标题主名: 取括号前的部分 (先剥离代码块, 避免 # 注释被误当标题)
    title = ""
    hm = H1_PATTERN.search(strip_code_blocks(content))
    if hm:
        title = hm.group(1).strip()

    norm = lambda s: re.sub(r"[\s\-_]+", " ", str(s).lower()).strip()

    # aliases: YAML 优先 (精确), 失败/异常时正则兜底 (兼容带/无引号)
    aliases = []
    try:
        fm2 = yaml.safe_load(fm_text)
        if isinstance(fm2, dict) and isinstance(fm2.get("aliases"), list):
            aliases = [str(a) for a in fm2["aliases"]]
    except Exception:
        pass
    if not aliases:
        aliases = extract_flow_list(fm_text, "aliases")

    tokens = set()
    for name in [pid, title, *aliases]:
        n = norm(name)
        if n:
            tokens.add(n)
    return {"id": pid, "tokens": tokens, "main_title": norm(title), "file": str(fp)}


def word_set(s: str) -> set:
    """id 拆词集 (按 _ - 空格)."""
    return set(re.split(r"[\s_\-]+", s.lower())) - {""}


def main():
    args = sys.argv[1:]
    include_all = "--all" in args
    dirs = SCOPE.items() if include_all else [("concepts", SCOPE["concepts"])]

    pages = []  # {id, tokens, main_title, file, kind}
    for key, (base, kind) in dirs:
        for fp in sorted(base.glob("*.md")):
            p = parse_page(fp)
            if p:
                p["kind"] = kind
                pages.append(p)

    if len(pages) < 2:
        print("✅ 页面不足, 无需检测")
        return 0

    candidates = []  # (strength, a, b, reason)
    for i in range(len(pages)):
        for j in range(i + 1, len(pages)):
            a, b = pages[i], pages[j]
            # 规则 1: token 直接重叠
            common = a["tokens"] & b["tokens"]
            if common:
                sample = sorted(common)[0]
                candidates.append((3, a, b, f"名称/别名重叠: {sample!r}"))
                continue
            # 规则 2: id 词集包含
            wa, wb = word_set(a["id"]), word_set(b["id"])
            if wa and wb and (wa < wb or wb < wa):
                short, long = (a, b) if wa < wb else (b, a)
                candidates.append((2, a, b,
                                   f"名称包含: {short['id']} ⊂ {long['id']}"))
                continue
            # 规则 3: 标题主名相同 (id 不同但标题一致)
            if a["main_title"] and a["main_title"] == b["main_title"]:
                candidates.append((1, a, b, "H1 标题主名相同"))

    if not candidates:
        print(f"✅ 无概念冲突候选 (扫描 {len(pages)} 页)")
        return 0

    candidates.sort(key=lambda c: (-c[0], c[1]["id"], c[2]["id"]))
    print(f"⚠️ {len(candidates)} 对候选, 需人工确认是否真为同术语不同定义:")
    for strength, a, b, reason in candidates:
        tag = {3: "强", 2: "中", 1: "弱"}[strength]
        print(f"  [{tag}] ({a['kind']}) {a['id']} ↔ ({b['kind']}) {b['id']}: {reason}")
        print(f"        {a['file']}")
        print(f"        {b['file']}")
    print("\n处置: 确认冲突 → 合并页面 (保留更深/更全/更实操的一篇); 不冲突 → 无需操作")
    return 1


if __name__ == "__main__":
    sys.exit(main())
