import os, re, yaml, json

sources_dir = 'wiki/sources'
raw_to_year = {}

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
    # Try arxiv_id: "1706.03762" -> YY=17 -> 2017
    arxiv = fm.get('arxiv_id', '')
    if arxiv:
        ym = re.match(r'^(\d{2})(\d{2})\.', str(arxiv))
        if ym:
            y = int(ym.group(1))
            year = 2000 + y
    
    # Try source id: "devlin_2018_bert" -> 2018
    if not year:
        sid = fm.get('id', '')
        ym = re.search(r'_(\d{4})_', sid)
        if ym:
            year = int(ym.group(1))
    
    # Find raw file path
    rm = re.search(r'\[阅读笔记\]\(\.\./\.\./raw/([^)]+)\)', content)
    if rm:
        raw_path = rm.group(1)
        raw_to_year[raw_path] = year

# Also add years for some files without wiki sources based on their names
# These are well-known papers with obvious years
extra_years = {
    # CNN classics
    "cnn/LeNet.md": 1998,
    "cnn/alexnet.md": 2012,
    "cnn/vgg.md": 2014,
    "cnn/googlenet.md": 2014,
    "cnn/inception_v3.md": 2015,
    "cnn/inception_v4.md": 2016,
    "cnn/SqueezeNet.md": 2016,
    "cnn/MobileNet_v1.md": 2017,
    "cnn/MobileNet_v2.md": 2018,
    "cnn/MobileNet_v3.md": 2019,
    "cnn/ShuffleNet.md": 2017,
    "cnn/ShuffleNet_v2.md": 2018,
    "cnn/EfficientNet.md": 2019,
    "cnn/regnet.md": 2020,
    "cnn/HarDNet.md": 2019,
    "cnn/GhostNet.md": 2020,
    "cnn/cspnet.md": 2019,
    "cnn/sparsenet.md": 2016,
    "cnn/ConvNeXt.md": 2022,
    "cnn/ConvNeXt_v2.md": 2023,
    "cnn/ResMLP.md": 2021,
    "cnn/repvgg.md": 2021,
    "cnn/Res2Net.md": 2019,
    "cnn/xception.md": 2016,
    "cnn/densenet.md": 2016,
    "cnn/resnext.md": 2016,
    # Detection
    "cnn/R-CNN.md": 2013,
    "cnn/Faster_R-CNN.md": 2015,
    "cnn/yolo_v1.md": 2015,
    "cnn/yolo_v2.md": 2016,
    "cnn/yolo_v3.md": 2018,
    "cnn/yolo_v4.md": 2020,
    "cnn/yolo_v6.md": 2022,
    "cnn/yolo_v7.md": 2022,
    "cnn/yolor.md": 2021,
    "cnn/yolox.md": 2021,
    "cnn/FCN.md": 2014,
    "cnn/pspnet.md": 2016,
    "cnn/DeepLab_v3.md": 2017,
    "cnn/panet.md": 2018,
    "cnn/blendmask.md": 2020,
    "cnn/rfb.md": 2018,
    "cnn/spach.md": 2022,
    "cnn/Non-local.md": 2017,
    "cnn/Deeply-Supervised.md": 2014,
    "cnn/Pooling.md": 2014,
    "cnn/BatchNorm.md": 2015,
    "cnn/GELUs.md": 2016,
    "cnn/Instance_Segmentation.md": 2017,
    "cnn/Semantic_Segmentation.md": 2017,
    # Self-supervised
    "cnn/MoCo.md": 2019,
    "cnn/MoCo_v2.md": 2019,
    "cnn/SimCLR.md": 2020,
    "cnn/DetCo.md": 2021,
    "cnn/cdp.md": 2020,
    "cnn/Consensus-Driven_Propagation.md": 2020,
    "cnn/MCUNet.md": 2020,
    "cnn/CNN_summary.md": 2020,
    # More NLP
    "nlp/gpt_4.md": 2023,
    "nlp/ALBERT.md": 2019,
    "nlp/BART.md": 2019,
    "nlp/Marian.md": 2019,
    "nlp/CoT-Multimodal.md": 2023,
    # Generative
    "Generative/DDPM.md": 2020,
    "Generative/LatentDiffusion.md": 2021,
    "Generative/Consistency_models.md": 2023,
    # RL
    "RL/DQN.md": 2013,
    "RL/DDPG.md": 2015,
    "RL/A2C.md": 2016,
    "RL/ACER.md": 2016,
    "RL/PPO.md": 2017,
    "RL/TRPO.md": 2015,
    "RL/hp_RL.md": 2019,
    # Root files
    "Dromedary.md": 2023,
    "LLaMA.md": 2023,
    "whisper.md": 2022,
    "self-Instruct.md": 2022,
    "RM_Overoptimization.md": 2022,
    "Xavier_init.md": 2010,
    "AdaLoRA.md": 2023,
    "LoRA.md": 2021,
    "QLoRA.md": 2023,
    "Deep_Compression.md": 2015,
    "model_compression.md": 2021,
    "Distilling_ss.md": 2015,
    "Switch_Transformers.md": 2021,
    "X-MoE.md": 2022,
    "MegaByte.md": 2023,
    "Vector_quantized.md": 2018,
    "FlashAttention.md": 2022,
    "diffusion.md": 2021,
    "OCR.md": 2012,
    "meta_learning_survey.md": 2017,
    "online_active_learning_survey.md": 2018,
    "DeepAL_survery_2009.00236.md": 2020,
    "DeepAL_survery_2203.13450.md": 2022,
    "mlp-mixer.md": 2021,
    "repmlp.md": 2021,
    "LayerNorm.md": 2016,
    # Autonomous Robot
    "Autonomous_Robot/DAVE-2.md": 2016,
    "Autonomous_Robot/Imitative_Models.md": 2020,
    "Autonomous_Robot/TransFuser.md": 2021,
    "Autonomous_Robot/LAV.md": 2021,
    "Autonomous_Robot/Label_Efficient.md": 2020,
    "Autonomous_Robot/ChauffeurNet.md": 2018,
    "Autonomous_Robot/Learning_Situational_Driving.md": 2018,
    "Autonomous_Robot/cheating.md": 2019,
    "Autonomous_Robot/H-Net.md": 2018,
    "Autonomous_Robot/E2E-LD.md": 2020,
    "Autonomous_Robot/Curve_LD.md": 2022,
    "Autonomous_Robot/DAgger.md": 2010,
    "Autonomous_Robot/limit_Behavior_Cloning.md": 2019,
    "Autonomous_Robot/RadarPerception.md": 2020,
    "Autonomous_Robot/VINS-Mono.md": 2017,
    "Autonomous_Robot/AdaRIP.md": 2022,
    "Autonomous_Robot/CAB.md": 2020,
    "Autonomous_Robot/Choice_data.md": 2022,
    "Autonomous_Robot/HiMODE.md": 2022,
    "Autonomous_Robot/MLDA.md": 2022,
    "Autonomous_Robot/mmTTransformer.md": 2022,
    "Autonomous_Robot/MUTR3D.md": 2022,
    "Autonomous_Robot/ONCE-3DLanes.md": 2022,
    "Autonomous_Robot/SAM.md": 2021,
    "Autonomous_Robot/Time3D.md": 2022,
    "Autonomous_Robot/TokenFusion.md": 2021,
    "Autonomous_Robot/UTT.md": 2021,
    "Autonomous_Robot/v2r_rl.md": 2021,
    "Autonomous_Robot/MoT_survey.md": 2022,
    # Video
    "video/MAE_st.md": 2022,
    "video/Review_video_prediction.md": 2020,
    "video/Unsupervised_Spatiotemporal.md": 2020,
}

raw_to_year.update(extra_years)

# Now build the reverse mapping: year -> list of raw files
year_groups = {}
no_year = []
for path, year in raw_to_year.items():
    if year:
        year_groups.setdefault(year, []).append(path)
    else:
        no_year.append(path)

# Output for debugging
print(f"Files WITH year: {len(raw_to_year)}")
print(f"Files WITHOUT year: {len(no_year)}")
print("\nYears covered:", sorted(year_groups.keys()))
for y in sorted(year_groups.keys()):
    print(f"  {y}: {len(year_groups[y])} files")

# Save mapping for later use
with open('/tmp/year_mapping.json', 'w') as f:
    json.dump(raw_to_year, f, ensure_ascii=False, indent=2)
