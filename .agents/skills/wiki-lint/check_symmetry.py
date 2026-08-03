#!/usr/bin/env python3
"""
check_symmetry: 验证 page 间 related_nodes 双向引用完整性.

原理: 如果 A 的 related_nodes 包含 B, 则 B 的 related_nodes 也应包含 A.

用法:
  python .agents/skills/wiki-lint/check_symmetry.py [page_id ...]
    不传参 → 检查所有 page
    传参   → 只检查指定 page (如 trend_following_drawdown_control)

退出码: 0 = 全部对称, 1 = 存在不对称, 2 = 指定的页面全部未找到
"""

import re
import sys
import pathlib
import yaml

_PROJ_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
WIKI_DIR = _PROJ_ROOT / "wiki"


def extract_id_field(fm_text: str) -> str | None:
    """从 frontmatter 原文正则提取 id 原始字符串 (避免 YAML float 化 / ParserError 整页跳过)."""
    lm = re.search(r"^id:\s*(.*)$", fm_text, re.MULTILINE)
    if not lm:
        return None
    raw = lm.group(1).strip()
    return raw.strip("'") or raw.strip('"') or None


def parse_page(path: pathlib.Path) -> dict:
    """读取一个 .md 文件, 返回 {id, related_nodes, filepath}. 失败返回 None."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None
    m = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not m:
        return None
    fm_text = m.group(1)
    pid = extract_id_field(fm_text)
    if not pid:
        return None

    ptype = ""
    related = set()
    # 正则优先 (单行格式): 避免纯数字 id (如 1404.5050) 被 YAML 浮点化; 兼容带/不带引号
    lm = re.search(r"^related_nodes:\s*\[([^\]]*)\]", fm_text, re.MULTILINE)
    if lm:
        for m in re.finditer(r"'([^']+)'|\"([^\"]+)\"|([^,\[\]\s]+)", lm.group(1)):
            item = next((g for g in m.groups() if g), "").strip().strip("'").strip('"')
            if item and item != "None":
                related.add(item)

    # YAML 兜底: 多行格式 / type 字段 / log 判定
    try:
        fm = yaml.safe_load(fm_text)
        if isinstance(fm, dict):
            ptype = str(fm.get("type", "")).strip()
            if not related:
                raw_rn = fm.get("related_nodes", []) or []
                for item in raw_rn:
                    s = str(item).strip().strip("'").strip('"')
                    if s and s != "None":
                        related.add(s)
    except Exception:
        pass

    # 跳过 log 文件（单向审计，不参与对称性校验）
    if ptype == "log":
        return None

    return {"id": pid, "related_nodes": related, "filepath": str(path.resolve()), "type": ptype}


def rn_items(fm_text: str) -> set:
    """从 frontmatter 原文提取 related_nodes 条目 (正则优先, 兼容带/无引号)."""
    out = set()
    lm = re.search(r"^related_nodes:\s*\[([^\]]*)\]", fm_text, re.MULTILINE)
    if lm:
        for m in re.finditer(r"'([^']+)'|\"([^\"]+)\"|([^,\[\]\s]+)", lm.group(1)):
            item = next((g for g in m.groups() if g), "").strip().strip("'").strip('"')
            if item and item != "None":
                out.add(item)
    return out


def auto_fix(by_id: dict) -> tuple:
    """--fix: 自动补 related_nodes.

    1. 给缺回引的页面补回引 (跳过 index/log 单向页)
    2. 清理幽灵引用 (related_nodes → 不存在的 id, 从源页移除)
    返回 (补回引数, 修改文件数).
    """
    fixes = {}  # target_id -> set(source_ids)
    ghost_src = {}  # source_id -> set(ghost_ids)
    for pid, page in by_id.items():
        if page["type"] in ("index", "log"):
            continue
        for target in page["related_nodes"]:
            tp = by_id.get(target)
            if tp is None:
                ghost_src.setdefault(pid, set()).add(target)
                continue
            if tp["type"] in ("index", "log"):
                continue
            if pid not in tp["related_nodes"]:
                fixes.setdefault(target, set()).add(pid)

    changed_files = 0

    # 1. 补回引
    for target, sources in fixes.items():
        fp = pathlib.Path(by_id[target]["filepath"])
        content = fp.read_text(encoding="utf-8")
        m = re.match(r"^(---\s*\n)(.*?)(\n---)", content, re.DOTALL)
        if not m:
            continue
        fm_text = m.group(2)
        lm = re.search(r"^(related_nodes:\s*\[)([^\]]*)(\])", fm_text, re.MULTILINE)
        if not lm:
            print(f"  [!] {target}: 无单行 related_nodes, 跳过")
            continue
        items = rn_items(fm_text)
        new_items = sorted(items | set(sources), key=lambda x: x.lower())
        if new_items == sorted(items, key=lambda x: x.lower()):
            continue
        new_line = f"related_nodes: [{', '.join(f"'{x}'" for x in new_items)}]"
        new_fm = fm_text[:lm.start(1)] + new_line + fm_text[lm.end(3):]
        fp.write_text(content[:m.start(2)] + new_fm + content[m.start(3):], encoding="utf-8")
        changed_files += 1

    # 2. 清理幽灵引用
    for pid, ghosts in ghost_src.items():
        fp = pathlib.Path(by_id[pid]["filepath"])
        content = fp.read_text(encoding="utf-8")
        m = re.match(r"^(---\s*\n)(.*?)(\n---)", content, re.DOTALL)
        if not m:
            continue
        fm_text = m.group(2)
        lm = re.search(r"^(related_nodes:\s*\[)([^\]]*)(\])", fm_text, re.MULTILINE)
        if not lm:
            continue
        items = rn_items(fm_text)
        new_items = sorted(items - set(ghosts), key=lambda x: x.lower())
        if new_items == sorted(items, key=lambda x: x.lower()):
            continue
        new_line = f"related_nodes: [{', '.join(f"'{x}'" for x in new_items)}]"
        new_fm = fm_text[:lm.start(1)] + new_line + fm_text[lm.end(3):]
        fp.write_text(content[:m.start(2)] + new_fm + content[m.start(3):], encoding="utf-8")
        print(f"  [幽灵清理] {pid}: 移除 {sorted(ghosts)}")
        changed_files += 1

    return len(fixes), changed_files


def build_graph(pages: list) -> dict:
    """构建 id → page dict, 以及反向引用表."""
    by_id = {}
    reverse = {}  # target_id → set of source_id
    for p in pages:
        by_id[p["id"]] = p
        for target in p["related_nodes"]:
            reverse.setdefault(target, set()).add(p["id"])
    return by_id, reverse


def check_page(pid: str, by_id: dict, reverse: dict) -> list:
    """检查单个 page 的对称性, 返回问题列表."""
    issues = []
    page = by_id.get(pid)
    if not page:
        return [f"  [!] '{pid}' 不存在或无法解析 frontmatter"]

    # 前向: 我引用了谁, 但对方没有回引我
    for target in page["related_nodes"]:
        if target not in by_id:
            issues.append(f"  [前向] '{pid}' → '{target}' 目标页面不存在")
            continue
        if pid not in by_id[target]["related_nodes"]:
            issues.append(f"  [前向] '{pid}' → '{target}' 缺少反向引用")

    # 反向: 谁引用了我, 但我没有回引对方
    incoming = reverse.get(pid, set())
    for source in incoming:
        if source not in by_id:
            continue
        # 跳过已在"前向"中报告的配对 (避免重复)
        if source not in page["related_nodes"]:
            if pid in by_id[source]["related_nodes"]:
                issues.append(f"  [反向] '{source}' → '{pid}' 但未回引")

    return issues


def main():
    args = sys.argv[1:]
    fix = "--fix" in args
    target_ids = [a for a in args if not a.startswith("--")]

    # 扫描所有 wiki 页面
    all_pages = []
    for fp in sorted(WIKI_DIR.rglob("*.md")):
        p = parse_page(fp)
        if p:
            all_pages.append(p)

    by_id, reverse = build_graph(all_pages)

    # --fix: 自动补回引 (跳过 index/log), 完成后退出
    if fix:
        n_missing, n_files = auto_fix(by_id)
        if n_missing == 0:
            print("✅ 无不齐, 无需修复")
        else:
            print(f"✅ 已补 {n_missing} 处回引 (修改 {n_files} 个文件, 跳过 index/log 单向页)")
        return 0

    if target_ids:
        to_check = [t for t in target_ids if t in by_id]
        not_found = [t for t in target_ids if t not in by_id]
        for t in not_found:
            print(f"[!] '{t}' 未找到")
    else:
        to_check = list(by_id.keys())
        not_found = []

    total_issues = 0
    total_index_issues = 0
    for pid in sorted(to_check):
        issues = check_page(pid, by_id, reverse)
        if issues:
            total_issues += len(issues)
            total_index_issues += sum(1 for l in issues if "'index'" in l)
            print(f"[{pid}]")
            for line in issues:
                print(line)

    if not_found and not to_check:
        print(f"\n⚠️ 指定的页面全部未找到")
        return 2
    if total_issues == 0:
        print(f"✅ 全部对称 (检查 {len(to_check)} 页)")
    else:
        print(f"\n⚠️ 共 {total_issues} 处不对称", end="")
        if total_index_issues:
            print(f"（其中 {total_index_issues} 处涉及 index 单向目录，属预期）")
        else:
            print()

    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
