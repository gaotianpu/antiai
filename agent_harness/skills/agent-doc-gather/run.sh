#!/usr/bin/env bash
# agent-doc-gather: 跨项目文档归集 — 扫描 workspace/agents，维护 file_mapping.md + 变更检测 + 版本对比
# 使用: bash <skill-dir>/run.sh  （不依赖 CWD）
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

README_FILE="$PROJECT_ROOT/README.md"

# === 步骤 1：扫描 ===
find ~/workspace -type f -path "*/rules/*" ! -path "$PROJECT_ROOT/*" 2>/dev/null | sort > /tmp/fm_workspace_rules.txt
find ~/.agents -type f 2>/dev/null | sort > /tmp/fm_agents.txt
find ~/workspace/.agents -type f 2>/dev/null | sort > /tmp/fm_workspace_agents.txt

echo "📁 workspace rules/: $(wc -l < /tmp/fm_workspace_rules.txt) 个文件"
echo "📁 ~/.agents/: $(wc -l < /tmp/fm_agents.txt) 个文件"
echo "📁 workspace/.agents/: $(wc -l < /tmp/fm_workspace_agents.txt) 个文件"

# === 步骤 2：解析 README ===

# 2a. rules 条目
awk '/^## rules\/.*项目约束与指南/,/^---/' "$README_FILE" \
  | grep -E '^\| `[a-zA-Z_]' \
  | sed -E 's/^\| `([^`]+)`.*/\1/' \
  | sort > /tmp/fm_rules_entries.txt
echo "📋 README rules 条目: $(wc -l < /tmp/fm_rules_entries.txt)"

# 2b. skills 条目（从 skills/ 各子表中提取 skill 名）
awk '/^## skills\/.*Skill 清单/,/^## 文件统计/' "$README_FILE" \
  | grep -E '^\| `[a-z][a-z-]+`' \
  | sed -E 's/^\| `([^`]+)`.*/\1/' \
  | sort > /tmp/fm_skills_entries.txt
echo "📋 README skills 条目: $(wc -l < /tmp/fm_skills_entries.txt)"

# === 步骤 3：分类扫描结果 ===
# 所有文件路径
cat /tmp/fm_workspace_rules.txt /tmp/fm_agents.txt /tmp/fm_workspace_agents.txt \
  | sort -u > /tmp/fm_all_files.txt

# 3a. 提取 rules 类（basename||path）
> /tmp/fm_rules_raw.txt
while IFS= read -r f; do
  case "$f" in
    *"/rules/"*) echo "$(basename "$f")||$f" >> /tmp/fm_rules_raw.txt ;;
  esac
done < /tmp/fm_all_files.txt

# 3b. 提取 skill 类（skill_name||path_to_SKILL.md）
> /tmp/fm_skills_raw.txt
while IFS= read -r f; do
  if [[ "$f" == */SKILL.md ]]; then
    # 提取 skill 目录名（SKILL.md 的父目录）
    skill_dir="$(basename "$(dirname "$f")")"
    echo "${skill_dir}||$f" >> /tmp/fm_skills_raw.txt
  fi
done < /tmp/fm_all_files.txt

# 3c. 其余文件 → orphans
> /tmp/fm_orphans_raw.txt
while IFS= read -r f; do
  case "$f" in
    *"/rules/"*) ;;
    */SKILL.md) ;;
    *) echo "$f" >> /tmp/fm_orphans_raw.txt ;;
  esac
done < /tmp/fm_all_files.txt

# === 步骤 4：构建 file_mapping.md ===
> /tmp/fm_mapping.md
echo "# File Mapping" > /tmp/fm_mapping.md
echo "" >> /tmp/fm_mapping.md
echo "> 最后更新: $(date '+%Y-%m-%d %H:%M:%S')" >> /tmp/fm_mapping.md
echo "" >> /tmp/fm_mapping.md

# --- 4a. Rules 映射 ---
echo "## Rules 文件" >> /tmp/fm_mapping.md
echo "" >> /tmp/fm_mapping.md

> /tmp/fm_matched_rules.txt
while IFS= read -r entry; do
  [ -z "$entry" ] && continue
  matches=$(grep "^${entry}||" /tmp/fm_rules_raw.txt | sed 's/^[^|]*||//' || true)
  if [ -n "$matches" ]; then
    echo "### $entry" >> /tmp/fm_mapping.md
    echo "" >> /tmp/fm_mapping.md
    echo "| # | 路径 |" >> /tmp/fm_mapping.md
    echo "|---|------|" >> /tmp/fm_mapping.md
    echo "$matches" | nl -ba -w1 -s' | ' | sed 's/^/| /' | sed 's/$/ |/' >> /tmp/fm_mapping.md
    echo "" >> /tmp/fm_mapping.md
    echo "$matches" >> /tmp/fm_matched_rules.txt
  fi
done < /tmp/fm_rules_entries.txt

# --- 4b. Skills 映射 ---
echo "## Skills" >> /tmp/fm_mapping.md
echo "" >> /tmp/fm_mapping.md

> /tmp/fm_known_skills.txt
> /tmp/fm_unknown_skills.txt
while IFS= read -r line; do
  [ -z "$line" ] && continue
  skill_name="${line%%||*}"
  skill_path="${line#*||}"
  
  if grep -qxF "$skill_name" /tmp/fm_skills_entries.txt 2>/dev/null; then
    echo "$line" >> /tmp/fm_known_skills.txt
  else
    echo "$line" >> /tmp/fm_unknown_skills.txt
  fi
done < /tmp/fm_skills_raw.txt

# 已知 skills
known_skills=$(cut -d"|" -f1 /tmp/fm_known_skills.txt | sort -u || true)
if [ -n "$known_skills" ]; then
  while IFS= read -r skill; do
    [ -z "$skill" ] && continue
    matches=$(grep "^${skill}||" /tmp/fm_known_skills.txt | sed 's/^[^|]*||//' || true)
    if [ -n "$matches" ]; then
      echo "### $skill/SKILL.md" >> /tmp/fm_mapping.md
      echo "" >> /tmp/fm_mapping.md
      echo "| # | 路径 |" >> /tmp/fm_mapping.md
      echo "|---|------|" >> /tmp/fm_mapping.md
      echo "$matches" | nl -ba -w1 -s' | ' | sed 's/^/| /' | sed 's/$/ |/' >> /tmp/fm_mapping.md
      echo "" >> /tmp/fm_mapping.md
    fi
  done <<< "$known_skills"
fi

# 未知 skills（不在 README 中的 skill）
unknown_skills=$(cut -d"|" -f1 /tmp/fm_unknown_skills.txt | sort -u || true)
if [ -n "$unknown_skills" ]; then
  echo "### ⚠️ 不属于 README 的 Skill" >> /tmp/fm_mapping.md
  echo "" >> /tmp/fm_mapping.md
  echo "| # | Skill | 路径 |" >> /tmp/fm_mapping.md
  echo "|---|-------|------|" >> /tmp/fm_mapping.md
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    skill_name="${line%%||*}"
    skill_path="${line#*||}"
    echo "  $line" >> /tmp/fm_unknown_full.txt
  done < /tmp/fm_unknown_skills.txt
  sort -u /tmp/fm_unknown_full.txt -o /tmp/fm_unknown_full.txt 2>/dev/null || true
  nl -ba -w1 -s' | ' /tmp/fm_unknown_full.txt 2>/dev/null \
    | sed 's/^/| /' | sed 's/||/ | /g' | sed 's/$/ |/' >> /tmp/fm_mapping.md
  echo "" >> /tmp/fm_mapping.md
  rm -f /tmp/fm_unknown_full.txt
  echo "📎 未在 README 中的 Skill: $(echo "$unknown_skills" | wc -l)"
fi

# --- 4c. 其余孤儿文件 ---
orphans=$(cat /tmp/fm_orphans_raw.txt | grep -v '^\s*$' || true)
if [ -n "$orphans" ]; then
  orphan_count=$(echo "$orphans" | sed '/^$/d' | wc -l)
  if [ "$orphan_count" -gt 0 ]; then
    echo "## 其他文件（非 rules 非 skill）" >> /tmp/fm_mapping.md
    echo "" >> /tmp/fm_mapping.md
    echo "| # | 路径 |" >> /tmp/fm_mapping.md
    echo "|---|------|" >> /tmp/fm_mapping.md
    echo "$orphans" | sed '/^$/d' | nl -ba -w1 -s' | ' | sed 's/^/| /' | sed 's/$/ |/' >> /tmp/fm_mapping.md
    echo "" >> /tmp/fm_mapping.md
  fi
fi

# === 步骤 5：报告 ===

# --- 5a. 其他项目有但本项目未收录的文档 ---
> /tmp/fm_missing_report.txt
> /tmp/fm_new_doc_candidates.txt

while IFS= read -r line; do
  [ -z "$line" ] && continue
  basename="${line%%||*}"
  # 检查本项目 rules/ 下是否有同名文件
  if [ ! -f "$PROJECT_ROOT/rules/$basename" ]; then
    echo "$line" >> /tmp/fm_new_doc_candidates.txt
  fi
done < /tmp/fm_rules_raw.txt

new_doc_count=$(sort -u /tmp/fm_new_doc_candidates.txt | wc -l)
if [ "$new_doc_count" -gt 0 ]; then
  echo "" >> /tmp/fm_mapping.md
  echo "## 报告：其他项目有但本项目未收录的文档" >> /tmp/fm_mapping.md
  echo "" >> /tmp/fm_mapping.md
  echo "以下文件存在于其他项目的 rules/ 中，但本项目 rules/ 下没有同名文件：" >> /tmp/fm_mapping.md
  echo "" >> /tmp/fm_mapping.md
  echo "| # | 文件名 | 来源路径 |" >> /tmp/fm_mapping.md
  echo "|---|--------|----------|" >> /tmp/fm_mapping.md
  sort -u /tmp/fm_new_doc_candidates.txt \
    | sed 's/||/ | /g' \
    | nl -ba -w1 -s' | ' \
    | sed 's/^/| /' \
    | sed 's/$/ |/' >> /tmp/fm_mapping.md
  echo "" >> /tmp/fm_mapping.md
fi

echo ""
echo "=== 报告 A：未收录的文档 ==="
if [ "$new_doc_count" -gt 0 ]; then
  echo "以下 $new_doc_count 个文件存在于其他项目，但本项目 rules/ 未收录："
  sort -u /tmp/fm_new_doc_candidates.txt | while IFS= read -r line; do
    fn="${line%%||*}"
    fp="${line#*||}"
    echo "  📄 $fn  ← $fp"
  done
else
  echo "✅ 其他项目无新增文档。"
fi

# --- 5b. 已收录文档的时间戳对比 ---
echo "" > /tmp/fm_staleness_report.txt
echo "=== 报告 B：版本新旧对比 ===" >> /tmp/fm_staleness_report.txt

has_stale=0
> /tmp/fm_stale_findings.txt

while IFS= read -r entry; do
  [ -z "$entry" ] && continue
  
  # 收集本项目 + 其他项目所有副本的路径
  local_file="$PROJECT_ROOT/rules/$entry"
  all_copies=""
  if [ -f "$local_file" ]; then
    all_copies="$local_file"
  fi
  remote_copies=$(grep "^${entry}||" /tmp/fm_rules_raw.txt | sed 's/^[^|]*||//' || true)
  if [ -n "$remote_copies" ]; then
    all_copies=$(printf "%s\n%s" "$all_copies" "$remote_copies" | sed '/^$/d')
  fi
  
  # 统计副本数
  copy_count=$(echo "$all_copies" | sed '/^$/d' | wc -l)
  [ "$copy_count" -le 1 ] && continue
  
  # 找最新的（按 mtime）
  newest_file=""
  newest_time=0
  while IFS= read -r cf; do
    [ -z "$cf" ] && continue
    [ ! -f "$cf" ] && continue
    mtime=$(stat -c '%Y' "$cf" 2>/dev/null || echo 0)
    if [ "$mtime" -gt "$newest_time" ]; then
      newest_time=$mtime
      newest_file="$cf"
    fi
  done <<< "$all_copies"
  
  # 判断最新版本是否是本项目
  if [ "$newest_file" != "$local_file" ] && [ -n "$newest_file" ]; then
    newest_date=$(date -d "@$newest_time" '+%Y-%m-%d %H:%M' 2>/dev/null || echo "unknown")
    echo "### $entry" >> /tmp/fm_stale_findings.txt
    echo "" >> /tmp/fm_stale_findings.txt
    echo "| 位置 | 路径 | 修改时间 |" >> /tmp/fm_stale_findings.txt
    echo "|------|------|----------|" >> /tmp/fm_stale_findings.txt
    while IFS= read -r cf; do
      [ -z "$cf" ] && continue
      [ ! -f "$cf" ] && continue
      mt=$(stat -c '%Y' "$cf" 2>/dev/null || echo 0)
      mt_date=$(date -d "@$mt" '+%Y-%m-%d %H:%M' 2>/dev/null || echo "unknown")
      label="其他"
      [ "$cf" = "$local_file" ] && label="本项目"
      [ "$cf" = "$newest_file" ] && label="🆕最新"
      echo "| $label | $cf | $mt_date |" >> /tmp/fm_stale_findings.txt
    done <<< "$all_copies"
    echo "" >> /tmp/fm_stale_findings.txt
    has_stale=1
  fi
done < /tmp/fm_rules_entries.txt

if [ "$has_stale" = "1" ]; then
  echo "" >> /tmp/fm_mapping.md
  echo "## 报告：版本新旧对比" >> /tmp/fm_mapping.md
  echo "" >> /tmp/fm_mapping.md
  echo "以下文档的最新版本不在本项目 rules/ 中：" >> /tmp/fm_mapping.md
  echo "" >> /tmp/fm_mapping.md
  cat /tmp/fm_stale_findings.txt >> /tmp/fm_mapping.md
fi

echo ""
echo "=== 报告 B：版本新旧对比 ==="
if [ "$has_stale" = "1" ]; then
  echo "以下文档的最新副本不在本项目："
  while IFS= read -r line; do
    case "$line" in
      "###"*) echo "${line### }" ;;
    esac
  done < /tmp/fm_stale_findings.txt
else
  echo "✅ 所有已收录文档的最新版本均在本项目。"
fi

# === 步骤 6：变更检测 ===
SNAPSHOT_DIR=~/.cache/fm_snapshot
SNAPSHOT_FILES="$SNAPSHOT_DIR/files"
mkdir -p "$SNAPSHOT_FILES"

has_changes=0
> /tmp/fm_changes.txt
> /tmp/fm_diffs.txt

while IFS= read -r f; do
  [ -z "$f" ] && continue; [ ! -f "$f" ] && continue
  safe_name=$(echo "$f" | sha256sum | cut -c1-16)
  hash=$(sha256sum "$f" | cut -d' ' -f1)
  
  if [ -f "$SNAPSHOT_DIR/$safe_name" ]; then
    old_hash=$(cat "$SNAPSHOT_DIR/$safe_name")
    if [ "$hash" != "$old_hash" ]; then
      echo "  🔴 已修改: $f" >> /tmp/fm_changes.txt
      echo "" >> /tmp/fm_diffs.txt
      echo "  🔴 --- $f" >> /tmp/fm_diffs.txt
      echo "  🔴 +++ $f (current)" >> /tmp/fm_diffs.txt
      if [ -f "$SNAPSHOT_FILES/$safe_name" ]; then
        diff -u "$SNAPSHOT_FILES/$safe_name" "$f" 2>/dev/null \
          | sed 's/^/  /' \
          | head -100 >> /tmp/fm_diffs.txt || true
      else
        echo "  (无历史副本，无法计算 diff)" >> /tmp/fm_diffs.txt
      fi
      has_changes=1
    fi
  else
    echo "  🟢 新增: $f" >> /tmp/fm_changes.txt
    has_changes=1
  fi
  echo "$hash" > "$SNAPSHOT_DIR/$safe_name"
  cp "$f" "$SNAPSHOT_FILES/$safe_name"
done < /tmp/fm_all_files.txt

# 检测已删除的快照
for snap in "$SNAPSHOT_DIR"/[!f]*; do
  [ -f "$snap" ] || continue
  snap_name=$(basename "$snap"); found=0
  while IFS= read -r f; do
    [ -z "$f" ] && continue; [ ! -f "$f" ] && continue
    [ "$(echo "$f" | sha256sum | cut -c1-16)" = "$snap_name" ] && { found=1; break; }
  done < /tmp/fm_all_files.txt
  if [ "$found" = "0" ]; then
    echo "  🗑️  已移除: $snap_name" >> /tmp/fm_changes.txt
    has_changes=1
    rm -f "$snap" "$SNAPSHOT_FILES/$snap_name" 2>/dev/null || true
  fi
done

# 生成 diff 报告文件
CHANGELOG_DIR="$PROJECT_ROOT/change_logs"
mkdir -p "$CHANGELOG_DIR"
if [ "$has_changes" = "1" ] && [ -s /tmp/fm_diffs.txt ]; then
  timestamp=$(date '+%Y%m%d_%H%M%S')
  report_file="$CHANGELOG_DIR/doc_diff_${timestamp}.md"
  {
    echo "# 文档变更报告"
    echo ""
    echo "> 生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "## 变更摘要"
    echo ""
    cat /tmp/fm_changes.txt
    echo ""
    echo "## Diff 详情"
    echo ""
    echo '```diff'
    # 去掉图标行，只保留 diff -u 原生内容
    cat /tmp/fm_diffs.txt | grep -v '🔴' | sed 's/^  //'
    echo '```'
    echo ""
    echo "---"
    echo "*由 agent-doc-gather 自动生成*"
  } > "$report_file"
  echo ""
  echo "📄 Diff 报告已保存: $report_file"
fi

echo ""
echo "=== 变更检测 ==="
if [ "$has_changes" = "1" ]; then
  cat /tmp/fm_changes.txt
  echo ""
  if [ -s /tmp/fm_diffs.txt ]; then
    echo "=== Diff 详情 ==="
    cat /tmp/fm_diffs.txt
    echo ""
    echo "📄 完整报告: $report_file"
  fi
else
  echo "无变更。"
fi

# === 写出 ===
cp /tmp/fm_mapping.md ./file_mapping.md
echo ""
echo "✅ 已生成 file_mapping.md ($(wc -l < ./file_mapping.md) 行)"

# === 可选：--apply 模式（将变更同步到本项目） ===
if [ "${1:-}" = "--apply" ] && [ "$has_changes" = "1" ]; then
  echo ""
  echo "=== 同步变更到本项目 ==="
  while IFS= read -r line; do
    case "$line" in
      "  🟢 新增: "*)
        src="${line#*  🟢 新增: }"
        ;;
      "  🔴 已修改: "*)
        src="${line#*  🔴 已修改: }"
        ;;
      *) continue ;;
    esac
    
    # 推导本项目对应路径
    # 规则: 将路径中的 workspace/xxx 或 .agents 映射到 PROJECT_ROOT
    case "$src" in
      *"/workspace/"*)
        # 从 workspace 路径提取文件相对路径
        rel="${src#*/workspace/}"      # project-name/path/to/file
        proj_name="${rel%%/*}"
        sub_path="${rel#*/}"            # path/to/file
        # 仅当不是本项目自身时才同步
        if [ "$PROJECT_ROOT" != "$HOME/workspace/$proj_name" ]; then
          dst="$PROJECT_ROOT/$sub_path"
          echo "  📋 同步: $src"
          echo "     → $dst"
          mkdir -p "$(dirname "$dst")"
          cp "$src" "$dst"
        fi
        ;;
      *"/.agents/"*)
        # 从 .agents 路径提取 skills/xxx/... 相对部分
        rel="${src#*/.agents/}"
        dst="$PROJECT_ROOT/$rel"
        echo "  📋 同步: $src"
        echo "     → $dst"
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
        ;;
    esac
  done < /tmp/fm_changes.txt
  echo "✅ 同步完成"
fi
