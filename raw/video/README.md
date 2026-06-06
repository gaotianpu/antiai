# 视频处理

## 一些问题
* 根据一段小的历史视频片段，预测下一帧图像，让模型学习真实世界的运动常识，有助于自动驾驶、自主机器人做规划决策。
* 视频数据必须是摄像头在真实物理世界中的采集，不能是影视、剪辑这种存在帧之间跳跃的视频类型; 
* 历史片段抽几帧，预测是紧跟着的下一帧，还是要间隔多少帧？
* 预测下一帧的具体目标是什么，真实与预测图像之间的diff哪些是重要的，哪些是不重要可忽略的？ 每个像素点的差异显然不合适，物体边框的IoU似乎又太依赖人工标注数据。MAE中每个块的像素均值方差？
* 预训练采用对比学习的思路：双塔模型接收历史视频片段+真正的下一帧图像(正)，非下一帧图像(负)，负样本选择是n秒后的帧？这样的预训练方案能学到的表征不够丰富？
* 反向预测：掩码掉历史视频中哪些像素，不影响下一帧的预测？

## 任务
1. 视频分类
2. 目标追踪 


## 一些文献
* 2022.9.18 [Masked Autoencoders As Spatiotemporal Learners](./MAE_st.md)
* 2021.4 [A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning](./Unsupervised_Spatiotemporal.md)
* 2020.4 [A Review on Deep Learning Techniques for Video Prediction](./Review_video_prediction.md)