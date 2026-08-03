---
id: visual_inertial_odometry
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [视觉惯性里程计, VIO]
related_nodes: [qin_2017_vinsmono, end_to_end_driving]
last_verified: 2026-08-03
---

# Visual-Inertial Odometry (VIO)

## 定义
融合相机与 IMU（惯性测量单元）估计运动轨迹的技术：视觉提供特征约束，IMU 提供尺度与短时积分，优势互补。

## 关键要点
- **互补性**：视觉慢漂移但无尺度问题（对单目），IMU 短时准但长时漂移
- **时间校准**：VINS-Mono 提出在线时间偏移校准，解决传感器异步
- **应用**：自动驾驶、AR、无人机定位

## 来源
- [[qin_2017_vinsmono]] — VINS-Mono
