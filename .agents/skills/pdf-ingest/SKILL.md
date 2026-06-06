---
name: pdf-ingest
description: PDF 下载→转换→格式清理，产出格式完好的 Markdown 文档。提示词：pdf转md, pdfmd, pdf2md,pdftomd
compatibility: Requires markitdown (pip install markitdown[pdf]) or pymupdf (pip install pymupdf), curl, and writable RAW_DIR/pdf/ directory.
metadata:
  variables:
    - RAW_DIR: 原始资料目录（默认 $PROJECT_ROOT/raw）
    - PROJECT_ROOT: 项目根目录（默认当前工作目录）
---

# PDF Ingest — Download → Convert → Cleanup

将 PDF（主要是 arXiv 论文）转换为格式完好的 Markdown，存入 `$RAW_DIR/`。

---

## 配置变量

| 变量 | 默认值 | 说明 |
|:---|:---|:---|
| `PROJECT_ROOT` | `$PWD` | 项目根目录 |
| `RAW_DIR` | `$PROJECT_ROOT/raw` | 原始资料目录 |

```bash
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
RAW_DIR="${RAW_DIR:-$PROJECT_ROOT/raw}"
```

---

> 以下流程假定 `ARXIV_ID` 已设置（arXiv 论文 ID，格式见 §1）。

## 0. Pre-flight Checks

```bash
which markitdown || echo "FATAL: markitdown not installed. Run: pip install markitdown[pdf]"
ls -d "$RAW_DIR/pdf/" || echo "FATAL: $RAW_DIR/pdf/ directory missing."
test -f "$RAW_DIR/${ARXIV_ID}.md" && echo "WARN: $RAW_DIR/${ARXIV_ID}.md already exists."
```

### 转换后立即检查空格完整性

```bash
# 若输出行中出现小写后紧跟大写（如 "aA"），说明空格丢失
grep -P '[a-z][A-Z]' "$RAW_DIR/${ARXIV_ID}.md" | head -5 && echo "WARN: spacing may be lost. Consider using pymupdf alternative (§2)."
```

---

## 1. Download PDF

```bash
ARXIV_ID="1601.00991"
curl -L --fail -o "$RAW_DIR/pdf/${ARXIV_ID}.pdf" "https://arxiv.org/pdf/${ARXIV_ID}"
```

- HTTP 404 → 检查 ID 格式（`YYYY.NNNNN`）
- 网络超时 → 重试最多 3 次，间隔 5s
- 非 arXiv 来源：替换 URL，其余流程不变

---

## 2. Convert to Markdown

```bash
markitdown "$RAW_DIR/pdf/${ARXIV_ID}.pdf" > "$RAW_DIR/${ARXIV_ID}.md"
```

### 备选工具：pymupdf

若 markitdown 输出单词粘连（如 `WhileMixture-of-Experts(MoE)scalescapacity`——空格丢失），改用 pymupdf：

```bash
pip install pymupdf
python3 -c "
import fitz
doc = fitz.open('$RAW_DIR/pdf/${ARXIV_ID}.pdf')
text = '\n'.join(page.get_text() for page in doc)
with open('$RAW_DIR/${ARXIV_ID}.md', 'w') as f:
    f.write(text)
"
```

> markitdown 将公式渲染为纯文本（非 LaTeX），如需 LaTeX 公式可尝试 `pandoc --from pdf --to markdown`。pymupdf 同理。

---

## 3. Format Cleanup

**必须先 git commit 原始转换结果**，确保后续修改可追溯。

### 3.1 文件大小检查

```bash
LINES=$(wc -l < "$RAW_DIR/${ARXIV_ID}.md")
[ "$LINES" -lt 15 ] && echo "FATAL: output too short (${LINES} lines). Conversion likely failed."
```

### 3.2 双栏乱序修复

**检测**：检查 `Abstract` 后是否紧跟 `1. Introduction`。若跳跃（如直接到 `1.2`），判定为双栏乱序。

```bash
grep -n "^[0-9]\." "$RAW_DIR/${ARXIV_ID}.md" | head -20   # 查看章节编号链
grep -n $'\f' "$RAW_DIR/${ARXIV_ID}.md"                     # 定位分页符
```

**修复**：
1. 移除 PDF 元数据 + 所有分页符 (`\f`)
2. 按章节编号重排文本块（1. → 1.1 → 1.2 …），确保 Introduction 在 Abstract 后
3. 策略：先对齐骨架（标题），再填充肌肉（正文），最后修饰皮肤（格式）

### 3.3 页眉/页脚/页码清除

```bash
grep -n '^[A-Z][a-z].\{2,40\}$' "$RAW_DIR/${ARXIV_ID}.md"   # 疑似页眉短行
grep -n '^[0-9]\+$' "$RAW_DIR/${ARXIV_ID}.md"               # 单独成行页码
```
人工确认后删除。

### 3.4 跨行断词修复

```bash
grep -n '[a-z]-$' "$RAW_DIR/${ARXIV_ID}.md"   # 行尾断字符
```
典型症状：`computa-` 行尾，下行以 `tion` 开头。合并为 `computation`。

### 3.5 标题层级修复

```bash
grep -n '^# ' "$RAW_DIR/${ARXIV_ID}.md"    # H1：必须恰好 1 个
grep -n '^## ' "$RAW_DIR/${ARXIV_ID}.md"   # H2：至少 3 个（Abstract + 正文 + References）
grep -n '^### ' "$RAW_DIR/${ARXIV_ID}.md"  # H3：可选
```

统一使用 `#` 层级，修复被切断的标题文字。原则上不应出现 `#####` 或更深层级，`####` 仅在论文确实有 4 级子节（如 6.1.1）时使用。

通用标题修复：

```bash
python tools/pdf2md_fix.py < "$RAW_DIR/${ARXIV_ID}.md" > tmp.md && mv tmp.md "$RAW_DIR/${ARXIV_ID}.md"
```

`tools/pdf2md_fix.py` 自动识别章节号并标注层级，修复 References/Abstract 标题，过滤引用条目误识别。后续发现可复用的转换修复逻辑追加到该文件。

### 3.6 代码块修复

识别破碎的代码行，合并为标准的 ```` ``` ```` 包裹块。确保语义完整。

### 3.7 表格修复

```bash
grep -n '^|' "$RAW_DIR/${ARXIV_ID}.md" | head -20    # 表格行
```

验证：表头分隔行 `|---|...` 存在、所有行列数一致、未被分页截断。将纯文本对齐表格转为 GFM `|` 格式。

### 3.8 LaTeX / 编码检查

```bash
grep -c '\begin{' "$RAW_DIR/${ARXIV_ID}.md"   # 应与 \end{ 数量相等
grep -c '\end{' "$RAW_DIR/${ARXIV_ID}.md"
grep -P '[\x80-\xFF]{3,}' "$RAW_DIR/${ARXIV_ID}.md" || echo "No encoding issues."
```

出现 `�` (U+FFFD) 或乱码 → 检查原始 PDF 是否含非英文内容。

### 3.9 参考文献确认

```bash
grep -n '^## References\|^## Bibliography\|^# References' "$RAW_DIR/${ARXIV_ID}.md"
```
必须命中 ≥1 行。缺失 → 检查 PDF 末尾截断。

### 3.10 质量校验清单

- [ ] 阅读逻辑是否符合原文意图
- [ ] 章节编号连续、标题层级正确
- [ ] 无断词、无页码/页眉残留
- [ ] 代码块语义完整、表格对齐
- [ ] 参考文献完整

---

## 4. Error Recovery

| 场景 | 诊断 | 对策 |
|:---|:---|:---|
| `curl` 下载失败 | 网络 / ID 格式 | 确认 ID 格式 `YYYY.NNNNN`；检查 `arxiv.org` 可达性 |
| `markitdown` 输出 < 15 行 | PDF 可能是扫描版（见 §3.1） | 放弃 markitdown，用 OCR 管线或标记 `TODO: OCR needed` |
| 公式严重变形 | 空格排版不可读（见 §2） | 尝试 `pandoc -f pdf -t markdown`；标注公式质量问题 |
| 双栏乱序 | 章节跳跃 | 按 §3.2 手动重排 |
| 输出空格丢失 | 单词粘连如 `aA` 模式 | 改用 pymupdf 重新提取（见 §2 备选工具） |
| 表格列错乱 | 复杂多列表格 | 手动修复；无法修复则标注 |
| PDF 含非英文内容 | 编码错误 | 检查 `iconv -l`，尝试指定编码重转 |
| 章节缺失 | 转换未识别 | 对比 PDF 目录，手动补全 `## ` 标题 |
