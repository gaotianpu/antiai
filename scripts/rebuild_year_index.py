"""
按年份正序重建 raw/index.md。

用法: python3 scripts/rebuild_year_index.py

流程:
  1. 从 wiki/sources/ 提取年份（arxiv_id 或 source id）
  2. 合并 extra_years（无 wiki source 的已知年份）
  3. 按年份正序（1990→最新）+ 主题分组写入 raw/index.md
"""

import os, re, yaml, json
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Phase 1: 提取年份映射 ──

raw_to_year = {}
sources_dir = os.path.join(PROJECT_ROOT, 'wiki/sources')

for fname in sorted(os.listdir(sources_dir)):
    if fname == 'index.md' or not fname.endswith('.md'):
        continue
    with open(os.path.join(sources_dir, fname)) as f:
        content = f.read()
    m = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not m:
        continue
    try:
        fm = yaml.safe_load(m.group(1))
    except:
        continue

    year = None
    arxiv = fm.get('arxiv_id', '')
    if arxiv:
        ym = re.match(r'^(\d{2})(\d{2})\.', str(arxiv))
        if ym:
            year = 2000 + int(ym.group(1))
    if not year:
        sid = fm.get('id', '')
        ym = re.search(r'_(\d{4})_', sid)
        if ym:
            year = int(ym.group(1))

    rm = re.search(r'\[阅读笔记\]\(\.\./\.\./raw/([^)]+)\)', content)
    if rm:
        raw_to_year[rm.group(1)] = year

# 文件已入库但无 阅读笔记 链接的，手动补
raw_to_year['2601.07372.md'] = 2026

# extra_years：无 wiki source 但有已知年份的文件
extra_years = {
    "nlp/gpt_4.md": 2023, "Generative/DDPM.md": 2020,
    "Generative/LatentDiffusion.md": 2021, "Generative/Consistency_models.md": 2023,
    "RL/DQN.md": 2013, "RL/DDPG.md": 2015, "RL/A2C.md": 2016, "RL/ACER.md": 2016,
    "RL/PPO.md": 2017, "RL/TRPO.md": 2015, "RL/hp_RL.md": 2019,
    "Dromedary.md": 2023, "LLaMA.md": 2023, "whisper.md": 2022,
    "self-Instruct.md": 2022, "RM_Overoptimization.md": 2022, "Xavier_init.md": 2010,
    "AdaLoRA.md": 2023, "LoRA.md": 2021, "QLoRA.md": 2023,
    "Deep_Compression.md": 2015, "model_compression.md": 2021, "Distilling_ss.md": 2015,
    "Switch_Transformers.md": 2021, "X-MoE.md": 2022, "MegaByte.md": 2023,
    "mlp-mixer.md": 2021, "repmlp.md": 2021,
    "meta_learning_survey.md": 2017, "online_active_learning_survey.md": 2018,
    "DeepAL_survery_2009.00236.md": 2020, "DeepAL_survery_2203.13450.md": 2022,
    "OCR.md": 2012, "diffusion.md": 2021,
    # Autonomous Robot
    "Autonomous_Robot/DAVE-2.md": 2016, "Autonomous_Robot/DAgger.md": 2010,
    "Autonomous_Robot/Imitative_Models.md": 2020, "Autonomous_Robot/TransFuser.md": 2021,
    "Autonomous_Robot/LAV.md": 2021, "Autonomous_Robot/Label_Efficient.md": 2020,
    "Autonomous_Robot/ChauffeurNet.md": 2018, "Autonomous_Robot/cheating.md": 2019,
    "Autonomous_Robot/H-Net.md": 2018, "Autonomous_Robot/Learning_Situational_Driving.md": 2018,
    "Autonomous_Robot/E2E-LD.md": 2020, "Autonomous_Robot/Curve_LD.md": 2022,
    "Autonomous_Robot/limit_Behavior_Cloning.md": 2019, "Autonomous_Robot/RadarPerception.md": 2020,
    "Autonomous_Robot/VINS-Mono.md": 2017, "Autonomous_Robot/CAB.md": 2020,
    "Autonomous_Robot/AdaRIP.md": 2022, "Autonomous_Robot/Choice_data.md": 2022,
    "Autonomous_Robot/HiMODE.md": 2022, "Autonomous_Robot/MLDA.md": 2022,
    "Autonomous_Robot/mmTTransformer.md": 2022, "Autonomous_Robot/MUTR3D.md": 2022,
    "Autonomous_Robot/ONCE-3DLanes.md": 2022, "Autonomous_Robot/SAM.md": 2021,
    "Autonomous_Robot/Time3D.md": 2022, "Autonomous_Robot/TokenFusion.md": 2021,
    "Autonomous_Robot/UTT.md": 2021, "Autonomous_Robot/v2r_rl.md": 2021,
    "Autonomous_Robot/MoT_survey.md": 2022, "Autonomous_Robot/SAM.md": 2021,
    # Video
    "video/MAE_st.md": 2022, "video/Review_video_prediction.md": 2020,
    "video/Unsupervised_Spatiotemporal.md": 2020,
}
raw_to_year.update(extra_years)

# ── Phase 2: 写入索引 ──

# 读取当前说明文本
index_path = os.path.join(PROJECT_ROOT, 'raw/index.md')
with open(index_path) as f:
    current = f.read()

raw_to_desc = {}
for line in current.split('\n'):
    if line.startswith('|') and '[[' in line:
        m = re.search(r'\[\[([^\]]+)\]\]\s*\|\s*(.+)', line)
        if m:
            desc = m.group(2).strip().rstrip('|').strip()
            raw_to_desc[m.group(1)] = desc

# 按年份分组
year_groups = defaultdict(list)
no_year = []
for path, year in raw_to_year.items():
    (year_groups[year] if year else no_year).append(path)

def get_topic(path):
    if path.startswith('cnn/'): return 'CNN'
    if path.startswith('nlp/'): return 'NLP'
    if path.startswith('vit/'): return 'ViT'
    if path.startswith('Multimodal/'): return '多模态'
    if path.startswith('Generative/'): return '生成模型'
    if path.startswith('RL/'): return 'RL'
    if path.startswith('Autonomous_Robot/'): return '自动驾驶'
    if path.startswith('video/'): return '视频'
    if path.split('/')[-1] in ('LoRA.md','QLoRA.md','AdaLoRA.md','Deep_Compression.md',
            'model_compression.md','Distilling_ss.md','Sparse_Expert_review.md',
            'Switch_Transformers.md','X-MoE.md','MegaByte.md','Vector_quantized.md',
            'FlashAttention.md','LayerNorm.md'):
        return '效率/压缩'
    return '其他'

topic_order = ['NLP', 'CNN', 'ViT', '多模态', '生成模型', 'RL', '自动驾驶', '视频', '效率/压缩', '其他']

out = ['# raw/ 索引', '', '按年份分组（正序）。']
for year in sorted(year_groups):
    files = year_groups[year]
    tg = defaultdict(list)
    for f in files:
        tg[get_topic(f)].append(f)
    out.append(f'\n## {year}')
    for topic in topic_order:
        if topic not in tg:
            continue
        out.append(f'\n### {topic}')
        for f in sorted(tg[topic]):
            desc = raw_to_desc.get(f, '')
            out.append(f'| [[{f}]] | {desc} |')

if no_year:
    out.append('\n## 年份不详')
    for f in sorted(no_year):
        desc = raw_to_desc.get(f, '')
        out.append(f'| [[{f}]] | {desc} |')

with open(index_path, 'w') as f:
    f.write('\n'.join(out) + '\n')

print(f"Done: {len(year_groups)} year groups, {sum(len(v) for v in year_groups.values())} files")
