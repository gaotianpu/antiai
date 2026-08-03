#!/usr/bin/env python3
"""
wiki-sync-index: 三向对齐 + 孤岛扫描 + 全库死链检测.

三个集合:
  r  = files on disk (实际文件)
  i1 = index.md 的 related_nodes YAML 字段
  i2 = index.md 正文中的 [[page_id]] 链接

比较:
  r \\ (i1 U i2)  → 文件存在但 index 完全未提
  i2 \\ i1        → 正文有但 related_nodes 漏了
  i1 \\ i2        → related_nodes 有但正文无显示
  (i1 U i2) \\ r  → index 提到但文件不存在

死链检测:
  代码块 (``` 围栏) 内的 [[...]] 视为代码字面量, 不参与.
  锚点链接 [[page#anchor]] 先整串匹配 all_ids (alpha101#1 是合法 id),
  不匹配再拆 # 验证 page 与 anchor 标题.

用法:
  python .agents/skills/wiki-lint/wiki-sync-index.py                  # 全部检查
  python .agents/skills/wiki-lint/wiki-sync-index.py --apply          # 检查并自动修正
  python .agents/skills/wiki-lint/wiki-sync-index.py --sync-only      # 只做 index 同步
  python .agents/skills/wiki-lint/wiki-sync-index.py --orphans-only   # 只做孤岛扫描
  python .agents/skills/wiki-lint/wiki-sync-index.py --deadlinks-only # 只做死链检测
  python .agents/skills/wiki-lint/wiki-sync-index.py --full           # 明细超过 30 条时不截断
  python .agents/skills/wiki-lint/wiki-sync-index.py --apply wiki/concepts/index.md

退出码: 0 = 全部对齐, 1 = 存在差异
"""

import re
import sys
import json
import pathlib
import yaml

_PROJ_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
WIKI_DIR = _PROJ_ROOT / "wiki"

# 顶层 index 是精选导航页 (非全集索引): 豁免 + 类 (文件未收录)
PLUS_EXEMPT = {"wiki/index.md"}

# 例外清单: 已记录/有意保留的问题, 不再报告 (格式: {deadlinks: [[源, 目标], ...], anchors: [[源, 目标], ...]})
EXCEPTIONS_FILE = pathlib.Path(__file__).resolve().parent / "exceptions.json"

INDEX_SCOPE = {
    "wiki/index.md":           (WIKI_DIR, "wiki/**/*.md"),
    "wiki/concepts/index.md":  (WIKI_DIR / "concepts", "*.md"),
    "wiki/sources/index.md":   (WIKI_DIR / "sources",  "*.md"),
    "wiki/entities/index.md":  (WIKI_DIR / "entities", "*.md"),
    "wiki/synthesis/index.md": (WIKI_DIR / "synthesis","*.md"),
}


# ═══════════════════════════════════════════
#  数据提取
# ═══════════════════════════════════════════

def parse_yaml_frontmatter(content: str):
    m = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not m:
        return None
    try:
        return yaml.safe_load(m.group(1))
    except Exception:
        return None


def frontmatter_block(content: str) -> str | None:
    """返回第一个 --- 块原文, 无则 None."""
    m = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    return m.group(1) if m else None


def extract_id_field(content: str) -> str | None:
    """从 frontmatter 原文正则提取 id 原始字符串.

    不经过 YAML 解析, 避免两个坑:
      1. 纯数字 id (1404.5050) 被 YAML 1.1 解析为 float, 尾 0 丢失 (→1404.505)
      2. aliases 等字段含 ' #xx' 注释导致 ParserError, 整页被跳过 (如 alpha101#N)
    """
    fm = frontmatter_block(content)
    if not fm:
        return None
    lm = re.search(r"^id:\s*(.*)$", fm, re.MULTILINE)
    if not lm:
        return None
    raw = lm.group(1).strip()
    return raw.strip("'\"") or None


def extract_type_field(content: str) -> str:
    """从 frontmatter 原文正则提取 type (YAML 解析失败时的兜底)."""
    fm = frontmatter_block(content)
    if not fm:
        return ""
    lm = re.search(r"^type:\s*(.*)$", fm, re.MULTILINE)
    if not lm:
        return ""
    return lm.group(1).strip().strip("'\"")


def load_exceptions() -> dict:
    """加载例外清单 (exceptions.json). 文件缺失/损坏时返回空."""
    if not EXCEPTIONS_FILE.exists():
        return {"deadlinks": [], "anchors": []}
    try:
        data = json.loads(EXCEPTIONS_FILE.read_text(encoding="utf-8"))
        return {"deadlinks": [tuple(x) for x in data.get("deadlinks", [])],
                "anchors": [tuple(x) for x in data.get("anchors", [])]}
    except Exception:
        return {"deadlinks": [], "anchors": []}


def h1_desc(content: str, max_len: int = 40) -> str:
    """从正文提取第一个 H1 作为 index 条目的一句话描述 (截断)."""
    body = strip_code_blocks(content)
    hm = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    if not hm:
        return ""
    desc = hm.group(1).strip()
    return desc[:max_len] + ("…" if len(desc) > max_len else "")


def extract_file_ids(base_dir: pathlib.Path, glob_pattern: str, skip_paths: list = None) -> set:
    skip_paths = skip_paths or []
    ids = set()
    if "/" in glob_pattern:
        files = sorted((WIKI_DIR.parent).glob(glob_pattern))
    else:
        files = sorted(base_dir.glob(glob_pattern))
    for fp in files:
        if str(fp.resolve()) in skip_paths:
            continue
        fid = extract_id_field(fp.read_text(encoding="utf-8"))
        if fid:
            ids.add(fid)
    return ids


def auto_add_entries(results: list, page_info: dict) -> bool:
    """--apply 扩展: 自动补 index 正文条目.
    +  类 (文件未收录, 顶层 index 豁免) → 追加 '- [[id]] — H1描述'
    ?  类 (related_nodes 有但正文无) → 追加 '- [[id]] — H1描述'
    返回是否有改动.
    """
    changed = False
    for result in results:
        rel = result["index"]
        r, i1, i2 = result["r"], result["i1"], result["i2"]
        to_add = (r - (i1 | i2)) if rel not in PLUS_EXEMPT else set()
        to_add |= i1 - i2  # ? 类: related_nodes 有但正文无
        if not to_add:
            continue
        path = WIKI_DIR.parent / rel
        content = path.read_text(encoding="utf-8")
        existing_links = extract_body_links(content)
        new_lines = []
        for pid in sorted(to_add):
            if pid in existing_links:
                continue
            info = page_info.get(pid)
            desc = h1_desc(info["content"]) if info else ""
            new_lines.append(f"- [[{pid}]]" + (f" — {desc}" if desc else ""))
        if not new_lines:
            continue
        if not content.endswith("\n"):
            content += "\n"
        content += "\n## 📌 自动收录\n\n" + "\n".join(new_lines) + "\n"
        path.write_text(content, encoding="utf-8")
        print(f"  [{rel}] ✅ 自动补 {len(new_lines)} 条 index 条目")
        changed = True
    return changed


def extract_related_nodes(content: str) -> set:
    """提取 frontmatter related_nodes. 正则优先 (单行格式, 避免纯数字 id 被 YAML 浮点化).\n\n    兼容带引号 ('a', "a") 与无引号 (a) 两种条目写法."""
    out = set()
    fm = frontmatter_block(content)
    if fm:
        lm = re.search(r"^related_nodes:\s*\[([^\]]*)\]", fm, re.MULTILINE)
        if lm:
            for m in re.finditer(r"'([^']+)'|\"([^\"]+)\"|([^,\[\]\s]+)", lm.group(1)):
                item = next((g for g in m.groups() if g), "").strip().strip("'").strip('"')
                if item and item != "None":
                    out.add(item)
    if out:
        return out
    # 多行格式 (related_nodes:\n  - a) 兜底
    fm2 = parse_yaml_frontmatter(content)
    if not isinstance(fm2, dict):
        return out
    raw = fm2.get("related_nodes", []) or []
    for item in raw:
        s = str(item).strip().strip("'").strip('"')
        if s and s != "None":
            out.add(s)
    return out


def strip_code_blocks(body: str) -> str:
    """剥离 ``` 围栏代码块与 `行内代码`: 其中的 [[...]] 是代码/示例字面量."""
    lines, in_code = [], False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            lines.append(line)
    body = "\n".join(lines)
    return re.sub(r"`[^`\n]*`", "", body)


def extract_body_links(content: str) -> set:
    """从正文提取 [[page_id]], 跳过含 / 的外部路径与 ``` 代码块."""
    m = re.match(r"^---\s*\n.*?\n---\s*\n", content, re.DOTALL)
    body = content[m.end():] if m else content
    body = strip_code_blocks(body)
    links = set()
    for match in re.finditer(r"\[\[([^\[\]]+?)(?:\|[^\[\]]*?)?\]\]", body):
        target = match.group(1).strip()
        if "/" in target:
            continue
        target = target.replace("\\", "").strip()
        if target:
            links.add(target)
    return links


def build_page_index():
    """扫描全部 wiki .md 文件，返回:
      all_ids: set[str]           — 所有页面的 id
      page_info: dict[str, dict]  — id → {filepath, type, content}
    """
    all_ids = set()
    page_info = {}
    for fp in sorted(WIKI_DIR.rglob("*.md")):
        content = fp.read_text(encoding="utf-8")
        pid = extract_id_field(content)
        if not pid:
            continue
        fm = parse_yaml_frontmatter(content)
        if isinstance(fm, dict):
            ptype = str(fm.get("type", "")).strip()
        else:
            ptype = extract_type_field(content)
        all_ids.add(pid)
        # 重复 id (如各 index.md 的 id 均为 'index'): 合并 content, 保证每个文件的链接都参与扫描
        if pid in page_info:
            page_info[pid]["content"] += "\n" + content
        else:
            page_info[pid] = {"filepath": str(fp.resolve()), "type": ptype, "content": content}
    return all_ids, page_info


def read_index(path: pathlib.Path) -> dict:
    if not path.exists():
        return {"related_nodes": set(), "body_links": set(), "content": ""}
    content = path.read_text(encoding="utf-8")
    return {
        "related_nodes": extract_related_nodes(content),
        "body_links": extract_body_links(content),
        "content": content,
    }


# ═══════════════════════════════════════════
#  更新
# ═══════════════════════════════════════════

def update_index(path: pathlib.Path, new_related: set) -> bool:
    content = path.read_text(encoding="utf-8")
    m = re.match(r"^(---\s*\n.*?\n---)", content, re.DOTALL)
    if not m:
        return False
    fm = parse_yaml_frontmatter(m.group(1))
    if not isinstance(fm, dict):
        return False

    sorted_nodes = sorted(new_related, key=lambda x: x.lower())
    new_list = [f"'{x}'" for x in sorted_nodes]
    fm["related_nodes"] = sorted_nodes

    new_fm_lines = ["---"]
    for key, value in fm.items():
        if key == "related_nodes":
            new_fm_lines.append(f"{key}: [{', '.join(new_list)}]")
        elif isinstance(value, list):
            # 保持 flow 格式 [a, b], 不拆块格式 (避免每次 apply 产生格式噪音)
            new_fm_lines.append(f"{key}: [{', '.join(str(v) for v in value)}]")
        else:
            new_fm_lines.append(f"{key}: {value}")
    new_fm_lines.append("---")

    body = content[m.end():]
    path.write_text("\n".join(new_fm_lines) + body, encoding="utf-8")
    return True


# ═══════════════════════════════════════════
#  Index 同步
# ═══════════════════════════════════════════

def check_index(rel_str: str, all_ids: set = None) -> dict:
    abs_path = WIKI_DIR.parent / rel_str
    scope = INDEX_SCOPE.get(rel_str)
    if not scope:
        base_dir = abs_path.parent
        subdir = base_dir.name
        scope = (base_dir, "wiki/**/*.md" if subdir == "wiki" else "*.md")
    base_dir, glob_pat = scope

    r = extract_file_ids(base_dir, glob_pat, skip_paths=[str(abs_path.resolve())])
    idx = read_index(abs_path)
    i1 = idx["related_nodes"]
    i2 = idx["body_links"]

    return {"index": rel_str, "r": r, "i1": i1, "i2": i2, "all_ids": all_ids or set()}


LIMIT_SHOW = 15    # 截断时每类明细显示条数
LIMIT_TRUNC = 30   # 单类明细超过该条数即截断


def limit_rows(rows: list, prefix: str, items: set, full: bool) -> None:
    """向 rows 追加明细, 超过 LIMIT_TRUNC 条时截断为前 LIMIT_SHOW 条 + 计数."""
    sorted_items = sorted(items)
    for i, x in enumerate(sorted_items):
        if not full and len(sorted_items) > LIMIT_TRUNC and i >= LIMIT_SHOW:
            rows.append(f"    … 共 {len(sorted_items)} 条 (加 --full 显示全部)")
            return
        rows.append(f"{prefix} {x}")


def format_index_diff(result: dict, full: bool = False) -> list:
    r, i1, i2 = result["r"], result["i1"], result["i2"]
    all_ids = result.get("all_ids", set())
    idx = result["index"]
    rows = []

    missing = r - (i1 | i2)
    if result["index"] in PLUS_EXEMPT:
        missing = set()  # 顶层 index 是精选导航, + 类豁免
    if missing:
        rows.append(f"  [{idx}]  r \\ (i1\u222ai2)  {len(missing)} 个文件未在 index 中:")
        limit_rows(rows, "    +", missing, full)

    body_not_in_yaml = i2 - i1
    yaml_not_in_body = i1 - i2
    ghost = (i1 | i2) - r
    # 跨目录引用: 全库存在但不在本目录 → 不强制进入本目录 index 的 related_nodes/正文
    cross_ids = ghost & all_ids
    body_not_in_yaml -= cross_ids
    yaml_not_in_body -= cross_ids

    if body_not_in_yaml:
        rows.append(f"  [{idx}]  i2 \\ i1  {len(body_not_in_yaml)} 个正文 [[链接]] 未在 related_nodes 中:")
        limit_rows(rows, "    \u2192", body_not_in_yaml, full)

    if yaml_not_in_body:
        rows.append(f"  [{idx}]  i1 \\ i2  {len(yaml_not_in_body)} 个 related_nodes 条目不在正文:")
        limit_rows(rows, "    ?", yaml_not_in_body, full)

    if ghost:
        real_ghost = ghost - cross_ids
        if real_ghost:
            rows.append(f"  [{idx}]  (i1\u222ai2) \\ r  {len(real_ghost)} 个条目在 index 中但无对应文件:")
            limit_rows(rows, "    -", real_ghost, full)
        if cross_ids:
            rows.append(f"  [{idx}]  \u00b7 {len(cross_ids)} 个跨目录引用 (文件存在但不在本目录):")
            limit_rows(rows, "    \u00b7", cross_ids, full)

    if not missing and not body_not_in_yaml and not yaml_not_in_body and not ghost:
        rows.append(f"  [{idx}] \u2705 完全对齐 (r={len(r)}, i1={len(i1)}, i2={len(i2)})")

    return rows


def apply_index_fix(result: dict) -> list:
    r, i1, i2 = result["r"], result["i1"], result["i2"]
    abs_path = WIKI_DIR.parent / result["index"]
    actions = []
    log = []

    add_to_yaml = i2 - i1
    new_i1 = i1 | add_to_yaml
    ghost = (new_i1 | i2) - r
    final_related = (new_i1 - ghost) | (i2 - ghost)

    if final_related == i1:
        return actions

    if update_index(abs_path, final_related):
        if add_to_yaml:
            actually_added = add_to_yaml - ghost  # 跨目录条目不进本目录 index
            if actually_added:
                log.append(f"+{len(actually_added)}(\u6b63\u6587\u2192frontmatter)")
        if ghost:
            removed = i1 - final_related
            if removed:
                log.append(f"-{len(removed)}(\u67e5\u65e0\u6587\u4ef6)")
            body_ghost = i2 & ghost
            if body_ghost:
                log.append(f"! \u6b63\u6587\u4ecd\u6709 {len(body_ghost)} \u4e2a\u6b7b\u94fe\u9700\u624b\u52a8\u6e05\u7406")
        actions.append(f"  [{result['index']}] \u2705 ({', '.join(log)})")
    return actions


# ═══════════════════════════════════════════
#  孤岛扫描
# ═══════════════════════════════════════════

def scan_orphans(all_ids: set, page_info: dict) -> list:
    """找出没有任何入链的页面（排除 index 和 log）。"""
    incoming = {pid: set() for pid in all_ids}

    for pid, info in page_info.items():
        for target in extract_body_links(info["content"]):
            if target in incoming:
                incoming[target].add(pid)

    orphans = []
    for pid in sorted(all_ids):
        info = page_info.get(pid)
        if not info:
            continue
        # 跳过 index 和 log 文件
        if info["type"] in ("index", "log"):
            continue
        if len(incoming[pid]) == 0:
            orphans.append(pid)

    return orphans


def report_orphans(orphans: list) -> list:
    if not orphans:
        return ["  [\u5b64\u5c9b] \u2705 \u6ca1\u6709\u5b64\u5c9b\u9875\u9762"]
    rows = [f"  [\u5b64\u5c9b]  {len(orphans)} \u4e2a\u9875\u9762\u65e0\u4efb\u4f55\u5165\u94fe:"]
    for pid in orphans:
        rows.append(f"    ! {pid}")
    return rows


# ═══════════════════════════════════════════
#  全库死链检测
# ═══════════════════════════════════════════

ANCHOR_RE = re.compile(r"^#{1,6}\s*")


def anchor_exists(content: str, anchor: str) -> bool:
    """校验 [[page#anchor]] 的锚点是否存在于目标页 (标题行 / **加粗** / 纯文本行)."""
    esc = re.escape(anchor)
    return bool(re.search(rf"(^#{{1,6}}\s*{esc}|\*\*{esc}|^{esc})", content, re.MULTILINE))


def scan_deadlinks(all_ids: set, page_info: dict) -> tuple:
    """扫描所有页面正文的 [[page_id]], 报告目标不存在的链接.

    锚点链接 [[page#anchor]] 先整串匹配 all_ids (如 alpha101#1 是合法 id),
    不匹配再拆 # 验证 page 与 anchor 标题.
    返回 (deadlinks, anchor_missing). exceptions.json 中的条目不报告.
    """
    exc = load_exceptions()
    exc_dead = {tuple(x) for x in exc["deadlinks"]}
    exc_anchor = {tuple(x) for x in exc["anchors"]}
    deadlinks = []       # 目标页不存在
    anchor_missing = []  # 目标页存在但锚点标题缺失
    for pid, info in page_info.items():
        for target in extract_body_links(info["content"]):
            if target in all_ids:
                continue
            if "#" in target:
                page, _, anchor = target.partition("#")
                if page in all_ids:
                    if anchor and not anchor_exists(page_info[page]["content"], anchor):
                        if (pid, target) not in exc_anchor:
                            anchor_missing.append((pid, target))
                    continue
            if (pid, target) not in exc_dead:
                deadlinks.append((pid, target))
    return deadlinks, anchor_missing


def report_deadlinks(deadlinks: list, anchor_missing: list) -> list:
    rows = []
    if deadlinks:
        rows.append(f"  [\u6b7b\u94fe]  {len(deadlinks)} \u5904 [[links]] \u76ee\u6807\u4e0d\u5b58\u5728:")
        rows.append("    (\u6e90\u9875\u9762 \u2192 \u4e0d\u5b58\u5728\u7684\u76ee\u6807)")
        for src, tgt in sorted(deadlinks):
            rows.append(f"    {src} \u2192 {tgt}")
    if anchor_missing:
        rows.append(f"  [\u951a\u70b9]  {len(anchor_missing)} \u5904 [[page#anchor]] \u951a\u70b9\u6807\u9898\u4e0d\u5b58\u5728:")
        for src, tgt in sorted(anchor_missing):
            rows.append(f"    {src} \u2192 {tgt}")
    if not deadlinks and not anchor_missing:
        rows.append("  [\u6b7b\u94fe] \u2705 \u6ca1\u6709\u6b7b\u94fe")
    return rows


# ═══════════════════════════════════════════
#  main
# ═══════════════════════════════════════════

def main():
    args = sys.argv[1:]
    apply = "--apply" in args
    sync_only = "--sync-only" in args
    orphans_only = "--orphans-only" in args
    deadlinks_only = "--deadlinks-only" in args
    full = "--full" in args

    specific_paths = [a for a in args if not a.startswith("--")]

    has_diff = False

    # ── Index 同步 ──
    if not orphans_only and not deadlinks_only:
        # 全库 id 用于区分跨目录引用与真 ghost
        global_all_ids, global_page_info = build_page_index()
        paths = specific_paths or list(INDEX_SCOPE.keys())
        index_results = []
        for rel_str in paths:
            if not (WIKI_DIR.parent / rel_str).exists():
                print(f"  [!] \u8def\u5f84\u4e0d\u5b58\u5728: {rel_str}")
                has_diff = True
                continue
            result = check_index(rel_str, global_all_ids)
            index_results.append(result)
            rows = format_index_diff(result, full)
            for row in rows:
                print(row)
                if "\u2705" not in row:
                    has_diff = True

        if apply:
            # 自动补 index 正文条目 (+ / ? 类), 之后重算以同步 related_nodes
            if auto_add_entries(index_results, global_page_info):
                index_results = [check_index(r["index"], global_all_ids) for r in index_results]
                has_diff = True

        if apply and has_diff:
            print("\n--- \u5f00\u59cb\u4fee\u6b63 index ---")
            total_actions = 0
            for result in index_results:
                acts = apply_index_fix(result)
                total_actions += len(acts)
                for a in acts:
                    print(a)
            if total_actions == 0:
                print("  \u2705 \u65e0\u53ef\u81ea\u52a8\u4fee\u590d\u9879 (\u4e0b\u5217\u5dee\u5f02\u5747\u9700\u624b\u52a8\u5904\u7f6e)")
            print("--- \u5b8c\u6210 ---")
            if total_actions == 0:
                print("\u26a0\ufe0f \u4ee5\u4e0b\u5dee\u5f02\u65e0\u6cd5\u81ea\u52a8\u4fee\u6b63\uff08\u9700\u624b\u52a8\u5904\u7406\uff09\uff1a")
                print("   r \\ (i1\u222ai2)  \u2192 \u9700\u5728 index \u6b63\u6587\u6dfb\u52a0 [[page]] \u6761\u76ee")
                print("   i1 \\ i2       \u2192 \u786e\u8ba4\u662f\u5426\u9700\u8981\u4fdd\u7559\u5728 related_nodes \u6216\u8865\u5145\u6b63\u6587")
                print("   \u6b63\u6587\u6b7b\u94fe       \u2192 \u79fb\u9664\u4e0d\u5b58\u5728\u7684 [[page]] \u6216\u8865\u5efa\u6587\u4ef6")
            else:
                print("\n--- \u4fee\u6b63\u540e\u68c0\u67e5 ---")
                still_dirty = False
                for result in index_results:
                    rows = format_index_diff(check_index(result["index"], global_all_ids), full)
                    for row in rows:
                        print(row)
                        if "\u2705" not in row:
                            still_dirty = True
                if still_dirty:
                    print("\u26a0\ufe0f \u4ee5\u4e0b\u5dee\u5f02\u65e0\u6cd5\u81ea\u52a8\u4fee\u6b63\uff08\u9700\u624b\u52a8\u5904\u7406\uff09\uff1a")
                    print("   r \\ (i1\u222ai2)  \u2192 \u9700\u5728 index \u6b63\u6587\u6dfb\u52a0 [[page]] \u6761\u76ee")
                    print("   i1 \\ i2       \u2192 \u786e\u8ba4\u662f\u5426\u9700\u8981\u4fdd\u7559\u5728 related_nodes \u6216\u8865\u5145\u6b63\u6587")
                    print("   \u6b63\u6587\u6b7b\u94fe       \u2192 \u79fb\u9664\u4e0d\u5b58\u5728\u7684 [[page]] \u6216\u8865\u5efa\u6587\u4ef6")

    # ── 建全库索引（孤岛 + 死链共用） ──
    if not sync_only and (not specific_paths or orphans_only or deadlinks_only):
        print()
        all_ids, page_info = build_page_index()

        if not deadlinks_only:
            orphans = scan_orphans(all_ids, page_info)
            rows = report_orphans(orphans)
            for row in rows:
                print(row)
                if "\u2705" not in row:
                    has_diff = True

        if not orphans_only:
            deadlinks, anchor_missing = scan_deadlinks(all_ids, page_info)
            rows = report_deadlinks(deadlinks, anchor_missing)
            for row in rows:
                print(row)
                if "\u2705" not in row:
                    has_diff = True

    return 0 if not has_diff else 1


if __name__ == "__main__":
    sys.exit(main())
