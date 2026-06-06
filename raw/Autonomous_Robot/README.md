# Autonomous Driving 自动驾驶相关

##份
* 2022 https://zhuanlan.zhihu.com/p/532823137
* 2021 https://zhuanlan.zhihu.com/p/393178588  学习、识别、检测和理解
* 2020 https://zhuanlan.zhihu.com/p/115608705
* 2019 https://zhuanlan.zhihu.com/p/59575613
* 2018 https://www.aminer.cn/research_report/5c35d0d85a237876dd7f1287?download=true&pathname=selfdriving.pdf 

https://zhuanlan.zhihu.com/p/339527655 入门

https://www.tesla.com/AI

## 一些设想
* 人眼的特点：1. 双目且间距固定; 2.可远近变焦、上下左右对焦(物理对焦、算法对焦); 3.可通过颈部(相机云台)旋转或身体移动来调整视角;  视觉以外的信号，听觉？
* 特斯拉的全视觉感知路线是否可行？ 视觉识别的潜力还没完全挖掘出来? 简单了解下基于纯视觉的端到端无人驾驶，似乎大家都不看好。咋说呢，已有论文还没牵涉到视觉领域的超大模型？ 根据gpt3这样的文本领域大模型经验，是不是视觉领域未来的方向也是搞足够大的模型，也能产生"涌现"效果，小模型搞不定的任务，一下子就都能搞定？例如OpenAI的CLIP模型。视觉领域基于掩码机制的无监督预训练MAE是现成的了，视频版的MAE也有人探索了，缺少一个像gpt那样的自回归模型？根据视频中一段连续帧图像，预测下一帧图像，从而能学习到物体移动规律？ 可能是因为预测效果不太好量化评估？ 图像的多任务训练：分类、目标检测、实例分割、语义分割，图文结合等？ 关注视觉领域的大模型+prompt机制？ 
* SLAM是Simultaneous localization and mapping缩写，意为“同步定位与建图”，主要用于解决机器人在未知环境运动时的定位与地图构建问题


## 一、感知
1. 传感器：摄像头、毫米波雷达、激光雷达; 道路协同(BEV)
* 已知摄像头的经纬度、高度、角度等数据？
* 毫米波雷达，激光雷达的数据格式，加载方式还比较陌生
    * [Exploiting Temporal Relations on Radar Perception for Autonomous Driving](./RadarPerception.md)
* 感知融合,多模态数据融合算法 传感器融合|Sensor Fusion
    * [Multimodal Token Fusion for Vision Transformers](./TokenFusion.md)

2. 车道检测 lane detection  车道线/可行驶区域
* 语义分割为2部分： 区分地面和地面之上的物体？
* 技术路线：语义分割、按行分类、曲线拟合、基于锚点
* 评估指标 [Towards Driving-Oriented Metric for Lane Detection Models](./E2E-LD.md) 
* 2022.3 [Rethinking Efficient Lane Detection via Curve Modeling](./Curve_LD.md) 曲线拟合
* [Multi-level Domain Adaptation for Lane Detection](./MLDA.md)
* [ONCE-3DLanes: Building Monocular 3D Lane Detection](./ONCE-3DLanes.md)

3. 目标检测 object detection 
* 车辆、人、其他，
* 3D目标检测？

4. 目标跟踪 object tracking
* 2019 [DEEP LEARNING IN VIDEO MULTI-OBJECT TRACKING: A SURVEY](./MoT_survey.md) 多目标跟踪调研
* [MUTR3D: A Multi-camera Tracking Framework via 3D-to-2D Queries](./MUTR3D.md)
* [Time3D: End-to-End Joint Monocular 3D Object Detection and Tracking for Autonomous Driving](./Time3D.md)
* [Unified Transformer Tracker for Object Tracking](./UTT.md)

5. 轨迹预测 Trajectory Prediction
* motion prediction 运动预测的一般思路
* 预测附近物体的下一步轨迹？ 基于特征的预测和基于提议的预测. 障碍物轨迹预测 基于概率，proposal-based 输入输出格式
* [Multimodal Motion Prediction with Stacked Transformers](./mmTTransformer.md)
* 2019.9 [Raising context awareness in motion forecasting](./CAB.md)

6. 深度估计 Depth Estimation 
* 双目深度估计, 立体匹配 stereo matching, 视差估计 disparity estimation
    * 极线校正, 输入=左右两图 输出=视差图, d 每个像素对应的视差值, 定摄像机的基线距离 b 和焦距 f bf/d
    * 2021.4 [H-Net: Unsupervised Attention-based Stereo Depth Estimation Leveraging](./H-Net.md) 
* 单目深度估计 https://zhuanlan.zhihu.com/p/56263560
    * 2022.4 [HiMODE: A Hybrid Monocular Omnidirectional Depth](./HiMODE.md) 混合单目全向深度
    * [Binary TTC: A Temporal Geofence for Autonomous Navigation](#) 用了摄像头在运动过程中，连续的帧间产生光流，进行场景深度估计  


7. BEV(Bird's Eye View 鸟瞰图)感知
* [Online Temporal Calibration for Monocular Visual-Inertial Systems](https://arxiv.org/abs/1808.00692)


## 二、规划 Interactive Planning
### 端到端自动驾驶
#### 1. 纯视觉
* 2020.? [Learning Situational Driving](./Learning_Situational_Driving.md)
* 2020.? [Exploring data aggregation in policy learning for vision-based urban autonomous driving](./DAgger.md)
* 2020.5 [Label efficient visual abstractions for autonomous driving](./Label_Efficient.md)
* 2019.12 [SAM: Squeeze-and-Mimic Networks for Conditional Visual Driving Policy Learning](./SAM.md)
* 2019.12 [Learning by cheating](./cheating.md)
* 2019.11 [End-to-end model-free reinforcement learning for urban driving using implicit affordances](https://arxiv.org/abs/1911.10868)
* 2019.4 [Exploring the limitations of behavior cloning for autonomous driving](./limit_Behavior_Cloning.md)
* 2016.4 [End to End Learning for Self-Driving Cars](./DAVE-2.md)


#### 2. 纯激光雷达 ???
* 2020.6.26 [Can autonomous vehicles identify, recover from, and adapt to distribution shifts?](./AdaRIP.md)
* 2018.10.15 [Deep imitative models for flexible inference, planning, and control](./Imitative_Models.md)

#### 3. 多模融合的端到端
* 2021.4.19 [Multi-Modal Fusion Transformer for End-to-End Autonomous Driving](./TransFuser.md)
* [Learning to Drive by Watching YouTube Videos: Action-Conditioned Contrastive Policy Pretraining](#)
* [Virtual to Real Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/1704.03952)
* [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/abs/1812.03079)


* imitation learning 模仿学习，从视觉观察中学习感觉运动驾驶

## 三、控制
* 自身状态感知：当前位置(gps),方向,上坡下坡(陀螺仪)，速度、加速度，转向角度
* [Online Temporal Calibration for Monocular Visual-Inertial Systems](./VINS-Mono.md) 单目视觉惯性系统的在线时间校准

## 四、仿真 Simulation
* CARLA 城市驾驶模拟器 https://carla.org/
* https://github.com/udacity/self-driving-car-sim

## 五、高精地图 Lanes Network
1. 3D重建， 将摄像头、毫米波雷达、激光雷达数据整合为3D模型？
1. 建图与定位
1. 自动标注 Autolabeling


## 其他
* Corner Case(Anomaly Detection)解决方案
* 多任务学习网络
* 基础设施 Infrastructure

