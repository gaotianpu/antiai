import os, re, json

with open('/tmp/year_mapping.json') as f:
    year_map = json.load(f)

# Fix bad entries
if 'nlp/CoT-Multimodal.md' in year_map:
    del year_map['nlp/CoT-Multimodal.md']
year_map['Multimodal/CoT-Multimodal.md'] = 2023
year_map['vit/iGPT.md'] = 2020
year_map['2601.07372.md'] = 2026

# Read current index to get description texts
with open('raw/index.md') as f:
    current = f.read()

raw_to_desc = {}
for line in current.split('\n'):
    if line.startswith('|') and '[[' in line:
        m = re.search(r'\[\[([^\]]+)\]\]\s*\|\s*(.+)', line)
        if m:
            fpath = m.group(1)
            desc = m.group(2).strip().rstrip('|').strip()
            raw_to_desc[fpath] = desc

# Group by year
year_groups = {}
no_year = []
for path, year in year_map.items():
    if year:
        year_groups.setdefault(year, []).append(path)
    else:
        no_year.append(path)

def get_topic(path):
    if path.startswith('cnn/'): return 'CNN'
    if path.startswith('nlp/'): return 'NLP'
    if path.startswith('vit/'): return 'ViT'
    if path.startswith('Multimodal/'): return '多模态'
    if path.startswith('Generative/'): return '生成模型'
    if path.startswith('RL/'): return 'RL'
    if path.startswith('Autonomous_Robot/'): return '自动驾驶'
    if path.startswith('video/'): return '视频'
    if path in ('LoRA.md','QLoRA.md','AdaLoRA.md','Deep_Compression.md','model_compression.md',
                'Distilling_ss.md','Sparse_Expert_review.md','Switch_Transformers.md',
                'X-MoE.md','MegaByte.md','Vector_quantized.md','FlashAttention.md','LayerNorm.md'):
        return '效率/压缩'
    return '其他'

# Ensure all files in year_map exist on disk
all_good = True
for path in year_map:
    if not os.path.exists('raw/' + path):
        print(f'MISSING: {path}')
        all_good = False
if all_good:
    print('All files exist')

out = ['# raw/ 索引', '', '按年份分组（倒序），年份不详的留在末尾。']
topic_order = ['NLP', 'CNN', 'ViT', '多模态', '生成模型', 'RL', '自动驾驶', '视频', '效率/压缩', '其他']

for year in sorted(year_groups.keys(), reverse=True):
    files = year_groups[year]
    topic_groups = {}
    for f in files:
        t = get_topic(f)
        topic_groups.setdefault(t, []).append(f)
    
    out.append(f'\n## {year}')
    for topic in topic_order:
        if topic not in topic_groups:
            continue
        out.append(f'\n### {topic}')
        for f in sorted(topic_groups[topic]):
            desc = raw_to_desc.get(f, '')
            out.append(f'| [[{f}]] | {desc} |')

if no_year:
    out.append('\n## 年份不详')
    for f in sorted(no_year):
        desc = raw_to_desc.get(f, '')
        out.append(f'| [[{f}]] | {desc} |')

result = '\n'.join(out) + '\n'
with open('raw/index.md', 'w') as f:
    f.write(result)

total = sum(len(v) for v in year_groups.values())
print(f"Done: {len(year_groups)} year groups, {total} files")
