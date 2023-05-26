# Exploiting Temporal Relations on Radar Perception for Autonomous Driving
利用雷达感知的时间关系进行自动驾驶 https://arxiv.org/abs/2204.01184

## Abstract
We consider the object recognition problem in autonomous driving using automotive radar sensors. Comparing to Lidar sensors, radar is cost-effective and robust in all-weather conditions for perception in autonomous driving. However, radar signals suffer from low angular resolution and precision in recognizing surrounding objects. To enhance the capacity of automotive radar, in this work, we exploit the temporal information from successive ego-centric bird-eye-view radar image frames for radar object recognition. We leverage the consistency of an object’s existence and attributes (size, orientation, etc.), and propose a temporal relational layer to explicitly model the relations between objects within successive radar images. In both object detection and multiple object tracking, we show the superiority of our method compared to several baseline approaches.

我们考虑了使用汽车雷达传感器的自动驾驶中的物体识别问题。与激光雷达传感器相比，雷达在全天候条件下对自动驾驶的感知具有成本效益和稳健性。然而，雷达信号在识别周围物体时具有低的角分辨率和精度。为了增强汽车雷达的能力，在这项工作中，我们利用连续的以自我为中心的鸟瞰雷达图像帧中的时间信息进行雷达目标识别。我们利用对象的存在和属性(大小、方向等)的一致性，并提出了一个时间关系层来明确地建模连续雷达图像中对象之间的关系。在目标检测和多目标跟踪中，与几种基线方法相比，我们显示了我们方法的优越性。

## 1. Introduction
Autonomous driving utilizes sensing technology for robust dynamic object perception, and sequentially uses the perception for reliable and safe vehicle decision-making [40]. Among various perception sensors, camera and Lidar are the two dominant ones exploited for surrounding object recognition. The camera provides semantically rich visual features of traffic scenarios, while Lidar provides high-resolution point clouds that can capture the reflection from objects. Compared with camera and Lidar, radar enjoys the following unique advantages when applied in automotive applications. Primarily operating at 77 GHz, radar transmits electromagnetic waves at a millimeter wavelength to estimate the range, velocity, and angle of objects. At such a wavelength, it can penetrate or diffract around tiny particles in conditions such as rain, fog, snow, and dust, and offer long-range perception in these adverse weather conditions [41]. In contrast, laser sent by Lidar at a much shorter wavelength may bounce off these tiny particles, which leads to a significantly reduced operating range. Compared with the camera, radar is also resilient to light conditions, e.g., night and sun glare. Furthermore, radar offers a cost-effective and reliable option to complement other sensors. For the cost of Lidar, according to an aggressive estimate by Luminar, is expected to be the range of $500 - $1000 [1]. In contrast, automotive radar is expected to be less than $100 in 2022 [10]. However, as a disadvantage of radar-assisted automotive perception, a high angular resolution in the azimuth and elevation domains are indispensable. In recent open-access automotive radar datasets, an azimuth resolution of 1◦ becomes available, while the elevation resolution is still lagging behind. With 1◦ azimuth resolution, semantic features for objects in a short range, e.g., corners and shapes, can be observed, while an object at far distances can still be blurred due to the cross-range resolution. In summary, the capability of localizing and identifying objects for radar is still falling behind from full-level autonomous driving.

自动驾驶利用传感技术实现稳健的动态物体感知，并依次使用感知进行可靠和安全的车辆决策[40]。在各种感知传感器中，相机和激光雷达是用于周围物体识别的两种主要传感器。相机提供了交通场景的语义丰富的视觉特征，而激光雷达提供了高分辨率的点云，可以捕捉物体的反射。与相机和激光雷达相比，雷达在汽车应用中具有以下独特优势。雷达主要工作在77GHz，发射毫米波长的电磁波，以估计物体的距离、速度和角度。在这样的波长下，它可以在雨、雾、雪和灰尘等条件下穿透或衍射微小颗粒，并在这些不利天气条件下提供远距离感知[41]。相比之下，激光雷达发出的波长短得多的激光可能会从这些微小颗粒上反弹，这导致工作范围明显缩小。与相机相比，雷达对光线条件(如夜间和阳光眩光)也有弹性。此外，雷达为补充其他传感器提供了一种经济高效且可靠的选择。根据Luminar的积极估计，激光雷达的成本预计在500-1000美元之间[1]。相比之下，预计2022年汽车雷达价格将低于100美元[10]。然而，作为雷达辅助汽车感知的一个缺点，方位角和仰角域中的高角度分辨率是必不可少的。在最近的开放存取汽车雷达数据集中，方位分辨率为1◦ 变为可用，而高程分辨率仍然落后。带1◦ 方位分辨率，可以观察到短距离内物体的语义特征，例如角和形状，而远距离的物体仍然可以由于跨距离分辨率而模糊。总之，雷达定位和识别目标的能力仍然落后于全自动驾驶。

Figure 1. Showcasing of two successive radar images and the corresponding camera recording from Radiate dataset [25]. From top to bottom, we display examples in the normal, foggy, and snowy weather. The bounding boxes are the ground-truth annotations of objects where its color implies the object ID. The plotted arrows show the consistency of the object’s appearance and attributes within a short time period, e.g., length, width, and orientation. 
图1。显示两个连续的雷达图像和来自Radiate数据集的相应摄像机记录[25]。从上到下，我们展示了正常、大雾和下雪天气的样本。边界框是对象的基本事实注释，其颜色表示对象ID。绘制的箭头显示了短时间内对象外观和属性的一致性，例如长度、宽度和方向。


Some recent efforts have been taken to leverage and enhance automotive radar for object recognition from an algorithmic perspective. [17] proposes a deep-learning approach using range-azimuth-doppler measurement. [20] detects objects via synchronous radar and Lidar signals. Similarly, [15, 36] exploit the multi-modal sensing fusion. Besides deep learning, Bayesian learning has also attempted to solve extended object tracking with radar point clouds [34, 37]. The above works mainly focus on multi-modal sensing fusion for robust perception [15, 20, 36]. Differently, in this paper, we take our attempt to enhance the perception only using radar information, which requires fewer perception resources and avoids a complicated synchronized process for signals among multi-modal sensors.

最近已经采取了一些措施，从算法角度利用和增强汽车雷达进行目标识别。[17] 提出了一种使用距离-方位-多普勒测量的深度学习方法。[20] 通过同步雷达和激光雷达信号检测的物体。类似地，[15，36]利用了多模态传感融合。除了深度学习，贝叶斯学习还试图解决雷达点云的扩展目标跟踪[34，37]。上述工作主要集中于用于稳健感知的多模态传感融合[15，20，36]。不同的是，在本文中，我们尝试仅使用雷达信息来增强感知，这需要更少的感知资源，并避免了多模态传感器之间复杂的信号同步过程。

In this paper, we consider ego-centric bird-eye-view radar point clouds presented in a Cartesian frame, where pixel values indicate the strength of reflections. We develop an approach to enhance radar perception using temporal information. Based on the observation in Fig. 1, we assume that the same objects detected by radar within successive frames are consistent and share almost the same attributes, such as the object’s existence, length, orientation, etc. As a result, the detection at one frame can be facilitated by a previous/future frame through object-level correlations. To compensate for the blurriness and low angular resolution raised by radar sensors, we involve temporality and incorporate customized temporal relational layers to explicitly handle the object-level relations across successive frames.

在本文中，我们考虑笛卡尔坐标系中呈现的以自我为中心的鸟瞰雷达点云，其中像素值表示反射强度。我们开发了一种利用时间信息增强雷达感知的方法。基于图1中的观察，我们假设雷达在连续帧内检测到的相同对象是一致的，并且共享几乎相同的属性，例如对象的存在、长度、方向等。因此，通过对象级别相关性，前一帧/后一帧可以促进一帧的检测。为了补偿雷达传感器带来的模糊性和低角度分辨率，我们涉及时间性，并结合定制的时间关系层，以明确处理连续帧之间的对象级关系。

The temporal relational layer takes feature vectors at the potential object’s centers and conducts a temporal as well as a self-attention over the object features which are wrapped with their locality. Colloquially, this layer links temporally similar objects and transmits their representations, and is akin to feature smoothing. Hence, temporal relational layers could insert the inductive bias from object temporal consistency. Afterward, the object heatmap (indicating the center of objects) and relevant attributes are inferred upon the updated feature representation from temporal relational layers.

时间关系层在潜在对象的中心获取特征向量，并对用其位置包裹的对象特征进行时间和自注意。通俗地说，该层链接时间上相似的对象并传输其表示，类似于特征平滑。因此，时间关系层可以插入来自对象时间一致性的归纳偏差。随后，基于来自时间关系层的更新的特征表示来推断对象热图(指示对象的中心)和相关属性。

In this work, we consider the object recognition problem using radar in autonomous driving, which is a crucial alternative sensing technology that owes unique advantages. We underline major contributions of our work as follows:
* We facilitate the radar perception with additional temporal information to compensate for the blurriness and low angular resolution raised by radar sensors.
* We design a customized temporal relational layer, where the networks are inserted with an inductive bias that the same object in successive frames should share consistent appearance and attributes.
* We evaluate our method in object detection and multiple object tracking on Radiate dataset. With the comprehensive comparison to baseline methods, we show the consistent improvements brought by our method.

在这项工作中，我们考虑了在自动驾驶中使用雷达的目标识别问题，这是一种具有独特优势的关键替代传感技术。我们强调我们工作的主要贡献如下：
* 我们利用额外的时间信息来促进雷达感知，以补偿雷达传感器所带来的模糊性和低角度分辨率。
* 我们设计了一个定制的时间关系层，在该层中，网络被插入了一个归纳偏差，即连续帧中的同一对象应该共享一致的外观和属性。
* 我们在Radiate数据集上评估了我们的目标检测和多目标跟踪方法。通过与基线方法的全面比较，我们显示了我们的方法带来的一致改进。

## 2. Radar Perception: Background
Automotive radar dominantly uses frequency modulated continuous waveform (FMCW) to detect objects and gener- (a) Transmitter (Tx) (b) Receiver (Rx)<br/>
Figure 2. FMCW-based automotive radar. ate point clouds over multiple physical domains. As shown in Fig. 2 (a), it transmits a sequence of FMCW pulses through one of its M transmitting antennas: \label {st} s_m(t) = \sum \limits _{q=0}^{Q-1} c_m(q) s_p\left (t-nT_{\text {PRI}} \right ) e^{j 2\pi f_c t}, (1) where m and q are the indices for transmitting antenna and pulse, TPRI is pulse repetition interval, fc is the carrier frequency (e.g., 79 GHz), and sp(t) is baseband FMCW waveform (shown as the sinusoids in Fig. 2 (a)).

An object at a range of R0 with a radial velocity vt and a far-field spatial angle (i.e. azimuth, elevation, or both) induces amplitude attenuation and phase modulation to the received FMCW signal at each of N receiver RF chains (including the low noise amplifier (LNA), local oscillator (LO), and analog-to-digital converter (ADC)) of Fig. 2 (b).

The induced modulation from the target is captured by the baseband signal processing block (including fast Fourier transforms (FFTs) over range, Doppler, and spatial domains) in Fig. 2 (b). All these processes lead to a multi-dimensional spectrum. With the constant false alarm rate (CFAR) detection step that compares the spectrum with an adaptive threshold, radar point clouds are generated in the range,

Doppler, azimuth, and elevation domains [4, 13, 30].

Considering the computing and cost constraints, automotive radar manufactures may define the radar point clouds in a subset of the full four dimensions. For instance, traditional automotive radar generates detection points in the rangeDoppler domain, whereas some produce the points in the range-Doppler-azimuth plane [21]. In Radiate dataset [25] considered in this paper, the radar point cloud is defined in the range-azimuth plane with a 360◦ field view. The resulting polar-coordinate point cloud is further transformed into an ego-centric Cartesian coordinate system, then a standard voxelization can convert the point cloud into an image.

## 3. Radar Perception with Temporality
We present our framework in Fig. 3. Corresponding to

Fig. 3 from top to bottom, in the subsequent sections, we introduce the temporal feature extraction from two successive frames, the temporal relational layers, the learning method, followed by the extension to multiple object tracking.

Notation We clarify the following notations. θ denotes the learnable parameters in neural networks, and for simplification, we unify the notations of parameters with θ for all modules. We use a bracket following a three-dimensional matrix to represent the feature gathering process at certain coordinates. Consider a feature representation Z ∈ RC×H×W with C, H, and W represent channel, height, and width, respectively. Let P represent a coordinate (x, y) or a set of two-dimensional coordinates {(x, y)}K with cardinality equal to K and x, y ∈ R. Z[P] means taking the feature at a coordinate system indicated by P along width and height dimensions, with the returned features in RC or RK×C .

### 3.1. Temporal Feature Extraction
Denote a single radar frame as I ∈ R1×H×W . We concatenate two successive radar images: a current frame and its previous frame, along the channel dimension to involve temporal information at the input level. The channelconcatenated temporal input image for the current and previous frames can be respectively written as Ic+p and

Ip+c ∈ R2×H×W . The order of ‘current’ c and ‘previous’ p in the subscript indicates the feature-concatenating order of these two frames. We obtain the feature representations for the two frames by forwarding the formulated inputs through a backbone neural network Fθ(·): Z_c \defeq \mathcal {F}_\theta (I_{c+p}),\ \ Z_p \defeq \mathcal {F}_\theta (I_{p+c}). (2)

The backbone network Fθ(·) is built in standard deep convolutional neural networks (e.g., ResNet), and model parameters are shared for processing two inputs Ip+c and Ic+p.

To jointly involve high-level semantics and low-level finer details in feature representations, we build skip connections between features at different scales in neural networks. Specifically, for one skip connection, we up-sample the pooled feature from a deep layer to align its size with the feature from previous shallow layers via bilinear interpolation. A list of operations including convolution, non-linear activation, and batch normalization are afterward applied to the up-sampled feature. Next, the up-sampled features are concatenated with those from shallow layers along the channel dimension. Three skip connections are inserted into the networks to drive the features embrace semantics at four different levels. The final feature representation from the backbone neural networks are resulted in Zc, Zp ∈ RC× Hs × Ws , where s is the down-sampling ratio over the spatial dimension. We add an illustrative figure in Appendix A.

### 3.2. Modeling Object Temporal Relations
We design a temporal relational layer to model the correlation and consistency between potential objects in successive frames. The temporal relational layer receives multiple feature vectors from the two frames with each vector representing a potential object in a radar image. We apply a filtering module Gθ pre-hm : RC× Hs × Ws → R1× Hs × Ws on features Zc and Zp to select top K potential object features for the relational modeling. The set of coordinates Pc for potential objects in Zc is obtained via the following equation: \label {eq:pos} P_c \defeq \{ (x, y)\mid \mathcal {G}_\theta ^{\text {pre-hm}}(Z_{c})_{xy} \geq [\mathcal {G}_\theta ^{\text {pre-hm}}(Z_{c})]_K\}, (3) where [Gθ pre-hm(Zc)]K is the K-th largest value in Gθ pre-hm(Zc) over the spatial space Hs × Ws , and the subscript xy denotes taking value at coordinate (x, y). Clearly, the cardinality of

Pc is |Pc| = K. By substituting Zp into Eq. (3), Pp for Zp can be obtained similarly. We do not include features from all coordinates into the temporal relational layer due to that the computational complexity of the subsequent attention mechanism grows quadratically towards the value K.

By taking the coordinate sets Pc and Pp into feature representations, we have the selective feature matrix as: \textbf {H}_c \defeq Z_c[P_c],\ \textbf {H}_p \defeq Z_p[P_p]. (4)

Sequentially, let Hc+p :=  Hc, Hp ⊤ ∈ R2K×C denote the matrix concatenation of top-K selected features in the two frames that forms the input to the temporal relational layer.

We supplement the positional encoding into feature vectors before passing Hc+p into the temporal relational layer.

The reason is that Convolutional neural networks do not encompass absolute positional information into output feature representation since CNNs enjoy the translational invariance property. However, the position is crucial in object temporal relations because objects at a certain spatial distance in two successive frames are more likely to be associated and would share similar object’s attributes. The spatial distance between the same object is conditional on the frame rate and vehicle’s motion, and can be learned through a data-driven approach.

Denote H pos c+p ∈ R2K×(C+Dpos) as the feature supplemented by the positional encoding via feature concatenation, where

Dpos is the dimension of positional encoding. Positional encoding is projected from the normalized 2D coordinate (x, y) that takes values in [0, 1] via linear mappings.

Figure 3. The framework of radar object recognition with temporality. Viewing from left to right, our method takes two consecutive radar frames and extracts the temporal feature from each frame. Then, we select features that could be potential objects and learn the temporal consistency between them. Finally, several regression objectives are conducted upon the updated features for training.

Having the formulations above, we have our main operation for modeling the relations across frames. For a single l-th temporal relational layer, we use a superscript l to denote the input feature and l + 1 to denote the output feature: \label {eq:att} \textbf {H}_{c+p}^{l+1} = \text {softmax}\left (\frac {\textbf {M} + q(\textbf {H}_{c+p}^{l, \text {pos}}) k(\textbf {H}_{c+p}^{l, \text {pos}})^\top }{\sqrt {d}}\right )v(\textbf {H}_{c+p}^{l}), (5) where q(·), k(·), and v(·) are linear transformation layers applied to features and are referred as, respectively, query, keys, and values. d is the dimension of query and keys and is used to scale the dot product between them. The masking matrix M ∈ R2K×2K is defined as: \textbf {M} \defeq \sigma \cdot \left (\begin {bmatrix} \textbf {1}_{K,K}, \textbf {0}_{K,K}\\ \textbf {0}_{K,K}, \textbf {1}_{K,K}\\\end {bmatrix} - \mathbbm {1}_{2K} \right ), (6) where 1K,K is the all-one matrix with size K × K, 0K,K is the all-zero matrix with size K × K, ✶2K is the identity matrix of size 2K, and σ is a negative constant which is set to −(1e+10) in our implementation to guarantee a near-zero value in the output through softmax. The diagonal matrices of 1K,K disable the attention between features from the same frame, while the off-diagonal matrices of 0K,K allow the cross-frame attention. Also, the identity matrix ✶2K unlocks the object self-attention. The logic behind self-attention is that the same object co-occurrence cannot always be guaranteed in successive frames since an object can move out of the scope, thereby self-attention is desirable when an object is missing in only one frame. Noticeably, the positional encoding is only attached to keys and query but not to values, so the output feature does not involve locality. Other technical details follows the design of Transformer [29], and here we omit the detailed descriptions for simplification.

After executing the object temporal attention across frames in Eq. (5), we sequentially apply a feed-forward function that consists of two linear layers, layer normalization, and shortcut on features. The relational modeling is built with multiple temporal relational layers with the identical design. At the end, we split the updated features Hl+1 c and

Hl+1 p from Hlc +1 +p and refill the feature vector to Zc and Zp in the corresponding spatial coordinates from Pc and Pp.

Regressions in the next subsection are conducted on top of the refilled feature representations.

Discussion The above feature operations share some similarities with Transformer [29]. Transformer is designed for language representation learning, intending to map the words into a similar latent representation if two words are sharing correlations among the training corpus, including the co-existence, word positions, and semantics. The multihead attention operations in the stacked architecture can be understood as smoothing over the feature of semantically similar words [6, 8, 14]. In our context, the feature of objects with an identical ID in successive frames should be correlated and share a similar latent representation. This is particularly crucial since the latent representation store all object-relevant attributes and will be used for the subsequent decoding purpose, as elaborated in Section 3.3. The smoothing over two feature vectors of the same object in successive frames satisfies our basic temporal consistency assumption, and can enhance the detection when the object information is partially lost in one frame due to the blurriness from radar.

### 3.3. Learning
We pick the object’s center coordinates from the heatmap, and learn its attributes (i.e. the width, length, orientation, and center coordinate offset) from feature representations through regression.

Heatmap To localize objects, the 2D coordinate of a peak value in the heatmap is considered as the center of an object.

The heatmap is obtained by a module Gθ hm : RC× Hs × Ws → R1× Hs × Ws followed by a sigmoid function. We generate the ground-truth heatmap by placing the 2D radial basis function (RBF) kernel on the center of every ground-truth object, while the parameter σ in the RBF kernel is set proportional to the object’s width and length. Considering the sparsity of objects in radar images, we use focal loss [16] to balance the regression of ground-truth centers and background, and drive the predicted heatmap to approximate the ground-truth heatmap. Let hi and ˆhi denote the ground-truth and predicted value at i-th coordinate, N the total number of values in the heatmap, we express the focal loss as: \begin {split} L_h \defeq & -\frac {1}{N}\sum _i \big (\mathds {1}_{h_i = 1}(1 - \hat {h}_i)^\alpha \log (\hat {h}_i) \\ &+ \mathds {1}_{h_i \neq 1}(1 - h_i)^\beta \hat {h}_i^\alpha \log (1 - \hat {h}_i) \big ), \end {split} (7) where α and β are hyper-parameters and are chosen empirically with 2 and 4, respectively, following the prior work [38].

The same loss function is conducted for Gθ pre-hm to rectify the feature selection of the relational modeling. During inference, a threshold is set on the heatmap to distinguish the object center from backgrounds. Non-maximum suppression is applied to avoid excessive bounding boxes.

Width & Length We predict the width and length of an oriented bounding box from the feature vector positioned at the center coordinate in the feature map through another regression head Gbθ : RC → R2 . Let Pgt k denote the coordinate (x, y) of the center of k-th ground-truth object, bk the ground-truth vector containing width and length of k-th object, and Z a unified notation for Zc and Zp. We have:

L_\text {b} \defeq \frac {1}{N}\sum _{k=1}^{N} \text {Smooth}_{L_1} \left (\lVert \mathcal {G}_\theta ^\text {b}(Z[P_{\text {gt}}^k]) - b^k \rVert \right ), (8) where the L1 smooth loss is defined as: \text {Smooth}_{L_1}(x) \defeq \begin {cases} 0.5x^2 & \text {if}\ |x|< 1\\ |x|-0.5 & \text {otherwise}.\\ \end {cases} (9)

Orientation All vehicles are presented with an orientation in the bird-eye-view image. An angle range in [0◦, 360◦) can be measured by the deviation between the object’s orientation and the boresight direction of the ego vehicle.

We regress the sine and cosine values of the angle ϑ via

Grθ : RC → R2: L_\text {r} \defeq \frac {1}{N}\sum _{k=1}^{N} \text {Smooth}_{L_1}(\lVert \mathcal {G}_\theta ^\text {r}(Z[P_{\text {gt}}^k]) - (\sin (\vartheta ), \cos (\vartheta )) \rVert ). (10)

During the inference stage, the orientation can be predicted by sin(ϑˆ) and cos(ϑˆ)) via arctan(sin(ϑˆ)/cos(ϑˆ)).

Offset Down sampling in the backbone networks could incur a center coordinate shift for every object. The center coordinates in the heatmap are integers while the true coordinates are likely to be off the heatmap grids due to the spatial down sampling. To compensate for the shift, we calculate a ground-truth offset for the k-th object as: o^k \defeq \left (\frac {c_x^k}{s} - \left [\frac {c_x^k}{s}\right ],\ \frac {c_y^k}{s} - \left [\frac {c_y^k}{s} \right ] \right ), (11) where ckx and cky is the k-th center coordinate, s is the down sampling ratio, and the bracket [·] is the rounding operation to an integer. Having Goθ : RC → R2 , the regression for center positional offset can be similarly expressed as:

L_\text {o} \defeq \frac {1}{N}\sum _{k=1}^{N} \text {Smooth}_{L_1}(\lVert \mathcal {G}_\theta ^\text {o}(Z[P_{\text {gt}}^k]) - o^k \rVert ). (12)

Training All above regression functions compose the final training objective by a linear combination: \min _\theta \ L \defeq L_h + L_b + L_r + L_o. (13)

We omit the balanced factors for each term for simplification.

For each training step, our training procedure calculates the loss L and does the backward for both the current and previous frame simultaneously. Standing at the current frame, objects in the current frame receives information from the past for object recognition. On the other hand, from the previous frame perspective, objects utilize the temporal information from the immediate future frame. Therefore, the optimization can be viewed as a bi-directional backwardforward training towards two successive frames. For now, we do not extend the current framework to multiple frames, since an intermediate frame do not have a proper concatenated order of input images for temporal feature extraction would reduce the training efficiency. (neither from past to future or nor from future to past) and

### 3.4. Extending to Multiple Object Tracking
Our framework can be easily extended to online multiple object tracking by adapting a similar tracking procedure as in [42]. For multiple object tracking, we add a regression head to the center feature vector to predict a 2D moving offset between the center of an object holding the same tracking ID in current and previous frames. We simply use

Euclidean distance to accomplish the association in tracking decoding. We defer a detailed illustration and algorithm for

Multiple Object Tracking to Appendix B.

## 4. Experiment
### 4.1. Experimental Setup
Dataset We use the radar dataset Radiate [25] in our experiments for the following reasons: (1) it contains high-

Table 1. Experimental results of object detection on Radiate dataset. TRL is the abbreviation of ‘temporal relational layer.’

Split: train good weather Split: train good and bad weather mAP@0.3 mAP@0.5 mAP@0.7 mAP@0.3 mAP@0.5 mAP@0.7

RetinaNet-OBB-ResNet18 52.50± 1.81 37.83± 1.82 8.46± 0.61 49.44± 1.32 31.57± 1.54 6.97± 1.24

RetinaNet-OBB-ResNet34 50.79± 3.10 35.61± 3.35 7.67± 1.71 48.09± 3.85 31.10± 3.37 6.93± 1.60

RetinaNet-OBB-ResNet34-T. 52.52± 4.68 37.30± 3.35 8.75± 1.50 42.95± 3.46 24.50± 3.72 3.98± 1.55

CenterPoint-OBB-EfficientNetB4 61.15± 1.23 51.43± 1.45 20.31± 1.73 54.97± 2.59 42.37± 2.14 13.15± 0.98

CenterPoint-OBB-ResNet18 58.69± 3.09 49.41± 2.94 19.02± 1.80 55.83± 3.28 44.48± 3.19 14.43± 2.56

CenterPoint-OBB-ResNet34 59.42± 1.92 50.17± 1.91 18.93± 1.46 53.92± 3.44 42.81± 3.04 13.43± 1.92

BBAVectors-ResNet18 59.38± 3.47 50.53± 2.07 19.72± 1.10 56.84± 3.45 45.43± 2.87 15.07± 1.76

BBAVectors-ResNet34 60.88± 1.79 51.26± 1.99 19.86± 1.36 55.87± 2.90 44.61± 2.57 14.67± 1.45

Ours-EfficientNetB4-w/o TRL 60.77± 0.97 50.93± 1.27 20.31± 1.73 54.97± 2.59 42.37± 2.14 13.15± 0.98

Ours-EfficientNetB4-w. TRL 61.59± 1.54 50.98± 1.52 17.91± 1.48 55.28± 2.32 43.05± 2.63 13.48± 2.01

Ours-ResNet18-w/o TRL 57.48± 4.82 47.90± 4.77 16.85± 2.98 55.64± 2.32 44.48± 2.76 15.10± 1.68

Ours-ResNet18-w. TRL 62.79± 2.01 53.11± 1.96 20.57± 1.47 58.87± 3.31 46.42± 3.24 15.59± 2.31

Ours-ResNet34-w/o TRL 60.98± 1.89 49.98± 2.28 18.89± 1.46 57.21± 3.76 45.93± 3.52 15.51± 2.71

Ours-ResNet34-w. TRL 63.63± 2.08 54.00± 2.16 21.08± 1.66 56.18± 4.27 43.98± 3.75 14.35± 2.15

Table 2. Comparison on object detection to [25]. Results of [25] are directly copied from the original paper. split: train good weather mAP@0.5

FasterRCNN-ResNet50 [25] 45.31

FasterRCNN-ResNet101 [25] 45.84

Ours-ResNet18-w. TRL 48.02

Ours-ResNet34-w. TRL 48.66 resolution radar images; (2) it provides well-annotated oriented bounding boxes with tracking IDs for objects; and (3) it records various real driving scenarios in adverse weather.

Radiate is consist of video sequences recorded in adverse weather including sun, night, rain, fog, and snow. The driving scenarios vary from the motorway to the urban. The data format radar images generated from point clouds, where pixel values indicate the strength of radar signal reflections.

Radiate adopts mechanically scanning Navtech CTS350-X radar, providing 360◦ high-resolution range-azimuth images at 4 Hz. Currently, the radar does not afford doppler or velocity information. The whole dataset has in total 61 sequences and we follow the official 3 splits: train in good weather (31 sequences, 22383 frames, only in good weather, sunny or overcast), train good and bad weather (12 sequences, 9749 frames, both good and bad weather conditions), and test (18 sequences, 11305 frames, all kinds of weather conditions).

We separately train models on the former two training sets and evaluate on the test set. Numerical results from both two splits are reported. We also comprehensively review other public radar datasets and discuss why currently they are not feasible for our experiments in Section 5.

Baseline We implement several detectors, which have been well demonstrated in visual object detection for comparison. These detectors include: Faster-RCNN [22],

RetinaNet [16], CenterPoint [43], and BBAVectors [38].

The comparison is conducted with different backbone networks [9, 27]. Traditional detectors are not designed for oriented objects. To make them fit the oriented object detection, we manually add an extra dimension on anchors or regression to predict the angle of the object’s orientation. We denote the adaptation as ‘OBB’ (oriented bounding box) by the end of detector’s names in Table 1. To highlight the benefit from temporal modeling, we add the temporal input to baselines where ’T.’ indicates the input with two successive frames and ’Ours-w/o TRL’ is architecturally equivalence to the CenterPoint model with temporal input. For multiple object tracking, we include CenterTrack [42] on oriented objects that use the same tracking heuristics with us for comparison.

Implementation We follow [25] and exclude pedestrians and groups of pedestrians from detection and tracking targets since only very few reflections are observed in these two kinds of objects. We also do not distinguish the object categories like [25] because there is no significant difference between vehicle categories presented by radar signals (e.g., truck and bus). Regarding the computation, operations related to oriented rectangles like the calculation of the overlapping of oriented bounding boxes are conducted in CPU using DOTA benchmark toolkit [33], while the rest part on deep neural networks is running on a single RTX 3090. For all numerical results in Table 1, we apply a center crop with size 256×256 upon input images and exclude the targets outside this scope. This helps us to conduct comprehensive evaluations using our computational resource and numbers are averaged over 10 random seeds. For results in Table 2 and 3, we keep the original resolution with size 1152×1152 to make a fair comparison to the results from [25]. We set the gap of frames between two successive frames to 3 for detection and 1 for tracking, the position dimension Dp to 64, the number of temporal relational layers to 2, the batch size to 64 for cropped images with a gradient accumulation to every 2 steps, the learning rate to 5e-4 and weight decay to 1e-2 for Adam optimizer with five training epochs.

We adopt mean Average Precision (mAP) with Intersection over Union (IoU) at 0.3, 0.5, and 0.7 for the evaluation of oriented object detection. For multiple object tracking, we adopt the series of MOT metrics [18] including MOTA,

MOTP, IDSW, Frag., MT and PT, but defer the descriptions to Appendix B due to the page limitation.

### 4.2. Result and Analysis
Detection We report detection results in Table 1 and 2.

Our method consistently achieves better results on both two training splits among different levels of IoU thresholds. Besides, the margin between the performance with or without tempporal relational layers further confirms the contribution from modeling the temporal object consistence in successive frames. Regarding the two training splits, intuitively, adding more weather conditions into training could enhance the robustness of detection and tracking, since the testing set contains various weather. However, for radar, there is no significant difference in the presentation of data among diverse weather. The margin between two training splits mainly comes from the margin of the number of training samples. Regarding the difference in image size, there is a slight performance drop when involving a larger scope for detection. The drop comes from the cross-range resolution, where further objects might suffer from a heavier blurriness.

Tracking We report results on multiple object tracking in Table 3, where our methods achieve better performance comparing to baseline. For the baseline method, CenterTrack also considers the temporal information by adding the heatmap of the previous frame and the previous image into input during the inference stage. They use the ground-truth heatmap for training and the predicted heatmap for inference. This kind of learning can work well for RGB video tracking since the detection is mostly accurate. However, the detection on radar cannot achieve such accuracy so far, and therefore breaking the alignment of the heatmap in training and inference. The tracking performance with or without temporal relational layers highlights the effectiveness of modeling temporal object-level relations.

Visualization We present visualization results in Fig. 4 on both object detection and multiple object tracking, and more visualizations are attached in Appendix C. We observe many predictions hit the annotations with a slight shift. Except the correct predictions, it is noticeable that our model brings some false positive predictions. However, when looking into these false positives, with a high probability, they will be a cluster of reflections inside the box that can be viewed as a ghost object. This may be the main reason for creating these false positives. Meanwhile, our model miss some objects in the outer space. The reflections of missed objects are drowning in the reflections of static surroundings due to the low angular resolution. How to enhance the detection on ghost objects and blurriness would be an interesting problem.

We add an experiment in Appendix D to analyze the best amount of selective features in temporal relational layers.

The empirical results guide the heuristic setting of K.

## 5. Related Work
Radar Perception in Autonomous Driving There is an increasing attention on the adoption of radar in autonomous driving. We review some recent work from both algorithmic and radar resource perspectives. The work [17] proposes a deep-learning approach for automotive radar object detection using range-azimuth-doppler measurement. [20] focus on sensor fusion and propose a method to incorporate synchronous radar and Lidar signals for object detection. [15, 36] also exploit the multi-modal sensing fusion in autonomous driving. Besides deep learning, Bayesian learning has also been used for extended object tracking using radar [34, 37]. Our work only leverages radar signals but enhances the recognition with the temporal consistency on objects, which has not been explored by previous works. We defer a short review of current radar dataset in Appendix E.

Detection with Temporality Consecutive video frames could provide spatial-temporal cues for object recognition. [32] leverage a feature bank that extends the time horizon for spatial-temporal action localization. [26] and [3] insert the object-level association from short or long temporal dependency into Faster-RCNN [22] to capture the spatial-temporal information in object detection. Other techniques such as video pixel flow or 3D convolutions [35, 44, 45] are applied for visually rich video sequences but too heavy and not efficient for radar images. Our work shares the same philosophy that using spatial-temporal object-level correlation along the time horizon. However, all studies mentioned above are focusing on RGB video data but not design for oriented objects.

The object’s size and scale may not be consistent if an object is approaching or leaving the scope of the camera. Differently, we put our emphasis on radar data in autonomous driving, where the bird-eye-view point cloud-based images provide significant object property comparing to RBG video data. We design an anchor-free one-stage detector with temporality, which is efficient and does not have to tackle the pre-defined anchor parameters. The center-based detector is suitable for the bird-eye-view presentation since there is no object overlap from this view, hence the central feature is fully exposed to represent an object. Moreover, we do not explore the long-range dependency but restrict the consistency in only one successive frame, since vehicles can move out of the scope if the timescale is too long and consequently no more temporal relation is available.

Table 3. Experimental results of multiple object tracking on Radiate dataset. TRL is the abbreviation of ‘temporal relational layer.’ split: train good weather MOTA↑ MOTP↑ IDSW↓ Frag.↓ MT↑ PT↑

CenterTrack-ResNet18 0.1301 0.7026 873 920 269 254

CenterTrack-ResNet34 0.1455 0.7005 802 831 282 279

Ours-ResNet-18-w/o TRL 0.3293 0.7135 513 593 151 324

Ours-ResNet-18-w. TRL 0.3359 0.7349 349 498 145 330

Ours-ResNet-34-w/o TRL 0.3569 0.7080 557 640 179 362

Ours-ResNet-34-w. TRL 0.3791 0.7188 474 527 219 332

Figure 4. Visualizations on radar perception on Radiate dataset. The upper two figures show the object detection while the lower four sets of successive visualizations show multiple object tracking. In detection, green bounding boxes are ground-truth annotations, while red are model predictions. In multiple object tracking, bounding boxes are model predictions, colors indicate the object IDs, and plotted arrows show the moving of objects. Regarding the figure source, the left detection figure is from night-1-4, while the right one is from rain-4-0.

From left to right and top to bottom, the tracking sequences are from city-7-0, rain-4-0, fog-6-0, and junction-1-10.

Multiple Object Tracking A well-established paradigm for visual multiple object tracking [18] is tracking-bydetection [11, 23, 28]. The detected object bounding boxes are provided by an external detector, then data association techniques based on object appearance or motion are applied to detection to associate identical objects among candidates in multiple consecutive frames. Recent developments in multiple object tracking convert detectors into tracking algorithms to jointly detect and track objects [7, 39, 42]. We follow the simple tracking rule that is purely based on the cost of euclidean distance [39, 42] to extend our framework to multiple object tracking. Differently, [39, 42] only stack frames at multiple time steps as input, while our networks explicitly consider the object-level consistency.

## 6. Conclusion
We studied the object recognition problem using radar in autonomous driving. We facilitated the radar perception with temporality from video frames based on the assumption that the same object within successive frames should be consistent and share almost the same attributes. We designed a framework inserted with temporal relational layers to explicitly model the object-level consistency. We showed the effectiveness of our method by experiments in object detection and multiple object tracking.

Acknowledgement The authors would like to thank Petros T. Boufounos, Toshiaki Koike-Akino, Hassan Mansour, and Philip V. Orlik for their helpful discussion.

## References

1. Alan Ohnsman. Luminar Surges On Plan To Supply Laser Sensors For Nvidia’s Self-Driving Car Platform, 2021. 1
2. Dan Barnes, Matthew Gadd, Paul Murcutt, Paul Newman, and Ingmar Posner. The oxford radar robotcar dataset: A radar extension to the oxford robotcar dataset. In 2020 IEEE International Conference on Robotics and Automation, pages 6433–6438, 2020. 12
3. Sara Beery, Guanhang Wu, Vivek Rathod, Ronny Votel, and Jonathan Huang. Context r-cnn: Long term temporal context for per-camera object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13075–13085, 2020. 7
4. I. Bilik, O. Longman, S. Villeval, and J. Tabrikian. The rise of radar for autonomous vehicles: Signal processing solutions and future research directions. IEEE Signal Processing Magazine, 36(5):20–31, Sep. 2019. 2
5. Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11621–11631, 2020. 11
6. Yihe Dong, Jean-Baptiste Cordonnier, and Andreas Loukas. Attention is not all you need: Pure attention loses rank doubly exponentially with depth. arXiv preprint arXiv:2103.03404, 2021. 4
7. Christoph Feichtenhofer, Axel Pinz, and Andrew Zisserman. Detect to track and track to detect. In Proceedings of the IEEE International Conference on Computer Vision, pages 3038–3046, 2017. 8
8. Chengyue Gong, Dilin Wang, Meng Li, Vikas Chandra, and Qiang Liu. Improve vision transformers training by suppressing over-smoothing. arXiv preprint arXiv:2104.12753, 2021. 4
9. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 6
10. Jessie Lin and Hana Hu. Digitimes Research: 79GHz to replace 24GHz for automotive millimeter-wave radar sensors, 2017. 1
11. Xiaolong Jiang, Peizhao Li, Yanjing Li, and Xiantong Zhen. Graph neural based end-to-end data association framework for online multiple-object tracking. arXiv preprint arXiv:1907.05315, 2019. 8
12. Giseop Kim, Yeong Sang Park, Younghun Cho, Jinyong Jeong, and Ayoung Kim. Mulran: Multimodal range dataset for urban place recognition. In 2020 IEEE International Conference on Robotics and Automation, pages 6246–6253, 2020. 12
13. J. Li and P. Stoica. MIMO Radar Signal Processing. John Wiley & Sons, 2008. 2
14. Peizhao Li, Jiuxiang Gu, Jason Kuen, Vlad I. Morariu, Handong Zhao, Rajiv Jain, Varun Manjunatha, and Hongfu Liu. Selfdoc: Self-supervised document representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5652–5660, June 2021. 4
15. Teck-Yian Lim, Amin Ansari, Bence Major, Daniel Fontijne, Michael Hamilton, Radhika Gowaikar, and Sundar Subramanian. Radar and camera early fusion for vehicle detection in advanced driver assistance systems. In Machine Learning for Autonomous Driving Workshop at the 33rd Conference on Neural Information Processing Systems, 2019. 2, 7
16. Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Doll´ar. Focal loss for dense object detection. In Proceedings of the IEEE International Conference on Computer Vision, pages 2980–2988, 2017. 5, 6
17. Bence Major, Daniel Fontijne, Amin Ansari, Ravi Teja Sukhavasi, Radhika Gowaikar, Michael Hamilton, Sean Lee, Slawomir Grzechnik, and Sundar Subramanian. Vehicle detection with automotive radar using deep learning on rangeazimuth-doppler tensors. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, 2019. 1, 7
18. Anton Milan, Laura Leal-Taix´e, Ian Reid, Stefan Roth, and Konrad Schindler. Mot16: A benchmark for multi-object tracking. arXiv preprint arXiv:1603.00831, 2016. 7, 8, 11
19. Arthur Ouaknine, Alasdair Newson, Julien Rebut, Florence Tupin, and Patrick P´erez. Carrada dataset: camera and automotive radar with range-angle-doppler annotations. In 2020 25th International Conference on Pattern Recognition, pages 5068–5075, 2021. 11
20. Kun Qian, Shilin Zhu, Xinyu Zhang, and Li Erran Li. Robust multimodal vehicle detection in foggy weather using complementary lidar and radar signals. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 444–453, 2021. 1, 2, 7
21. Karthik Ramasubramanian and Brian Ginsburg. AWR1243 sensor: Highly integrated 76–81-GHz radar front-end for emerging ADAS applications. In Texas Instruments Technical Report, pages 1–12, 2017. 2
22. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in Neural Information Processing Systems, 28:91–99, 2015. 6, 7
23. Samuel Schulter, Paul Vernaza, Wongun Choi, and Manmohan Chandraker. Deep network flow for multi-object tracking. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6951–6960, 2017. 8
24. Ole Schumann, Markus Hahn, Nicolas Scheiner, Fabio Weishaupt, Julius F Tilly, J¨urgen Dickmann, and Christian W¨ohler. Radarscenes: A real-world radar point cloud data set for automotive applications. arXiv preprint arXiv:2104.02493, 2021. 11
25. Marcel Sheeny, Emanuele De Pellegrin, Saptarshi Mukherjee, Alireza Ahrabian, Sen Wang, and Andrew Wallace. Radiate: A radar dataset for automotive perception. arXiv preprint arXiv:2010.09076, 2020. 1, 2, 5, 6, 12
26. Mykhailo Shvets, Wei Liu, and Alexander C Berg. Leveraging long-range temporal relationships between proposals for video object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9756–9764, 2019. 7
27. Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International Conference on Machine Learning, pages 6105–6114. PMLR, 2019. 6
28. Siyu Tang, Mykhaylo Andriluka, Bjoern Andres, and Bernt Schiele. Multiple people tracking by lifted multicut and person re-identification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3539– 3548, 2017. 8
29. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008, 2017. 4
30. Pu Wang, Petros Boufounos, Hassan Mansour, and Philip V. Orlik. Slow-time MIMO-FMCW automotive radar detection with imperfect waveform separation. In 2020 IEEE International Conference on Acoustics, Speech and Signal Processing, pages 8634–8638, 2020. 2
31. Yizhou Wang, Gaoang Wang, Hung-Min Hsu, Hui Liu, and Jenq-Neng Hwang. Rethinking of radar’s role: A cameraradar dataset and systematic annotator via coordinate alignment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2815–2824, 2021. 11
32. Chao-Yuan Wu, Christoph Feichtenhofer, Haoqi Fan, Kaiming He, Philipp Krahenbuhl, and Ross Girshick. Long-term feature banks for detailed video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 284–293, 2019. 7
33. Gui-Song Xia, Xiang Bai, Jian Ding, Zhen Zhu, Serge Belongie, Jiebo Luo, Mihai Datcu, Marcello Pelillo, and Liangpei Zhang. Dota: A large-scale dataset for object detection in aerial images. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3974–3983, 2018. 6
34. Yuxuan Xia, Pu Wang, Karl Berntorp, Lennart Svensson, Karl Granstr¨om, Hassan Mansour, Petros Boufounos, and Philip V Orlik. Learning-based extended object tracking using hierarchical truncation measurement model with automotive radar. IEEE Journal of Selected Topics in Signal Processing, 15(4):1013–1029, 2021. 2, 7
35. Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu, and Kevin Murphy. Rethinking spatiotemporal feature learning: Speed-accuracy trade-offs in video classification. In Proceedings of the European Conference on Computer Vision, pages 305–321, 2018. 7
36. Bin Yang, Runsheng Guo, Ming Liang, Sergio Casas, and Raquel Urtasun. Radarnet: Exploiting radar for robust perception of dynamic objects. In European Conference on Computer Vision, pages 496–512, 2020. 2, 7
37. Gang Yao, Perry Wang, Karl Berntorp, Hassan Mansour, P Boufounos, and Philip V Orlik. Extended object tracking with automotive radar using b-spline chained ellipses model. In 2021 IEEE International Conference on Acoustics, Speech and Signal Processing, pages 8408–8412, 2021. 2, 7
38. Jingru Yi, Pengxiang Wu, Bo Liu, Qiaoying Huang, Hui Qu, and Dimitris Metaxas. Oriented object detection in aerial images with box boundary-aware vectors. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2150–2159, 2021. 5, 6
39. Tianwei Yin, Xingyi Zhou, and Philipp Krahenbuhl. Centerbased 3d object detection and tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11784–11793, 2021. 8
40. Ekim Yurtsever, Jacob Lambert, Alexander Carballo, and Kazuya Takeda. A survey of autonomous driving: Common practices and emerging technologies. IEEE access, 8:58443– 58469, 2020. 1
41. Shuqing Zeng and James N. Nickolaou. Automotive radar. In Gregory L. Charvat, editor, Small and Short-Range Radar Systems, chapter 9. CRC Press, Inc., 2014. 1
42. Xingyi Zhou, Vladlen Koltun, and Philipp Kr¨ahenb¨uhl. Tracking objects as points. In European Conference on Computer Vision, pages 474–490, 2020. 5, 6, 8, 11
43. Xingyi Zhou, Dequan Wang, and Philipp Kr¨ahenb¨uhl. Objects as points. arXiv preprint arXiv:1904.07850, 2019. 6
44. Xizhou Zhu, Yujie Wang, Jifeng Dai, Lu Yuan, and Yichen Wei. Flow-guided feature aggregation for video object detection. In Proceedings of the IEEE International Conference on Computer Vision, pages 408–417, 2017. 7
45. Xizhou Zhu, Yuwen Xiong, Jifeng Dai, Lu Yuan, and Yichen Wei. Deep feature flow for video recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2349–2358, 2017. 7 

## A. Temproal Feature Extraction

We add Fig. 5 to illustrate the skip connections in the backbone neural networks. Skip connections within CNNs are designed to jointly involve high-level semantics and lowlevel finer details in output feature representation. Specially, we add three skip connections in ResNet and gradually upsample the features from a deeper layer. The final feature representations are down-sampled with a ratio of 4 compared to the original inputs in this U-Net structure.

## B. Multiple Object Tracking: Evaluation and the Decoding Algorithm

We adopt the series of MOT metrics [18] for evaluation.

We pick several key metrics in experiments: MOTA (Multiple Object Tracking Accuracy), MOTP (Multiple Object

Tracking Precision), ID switch (IDSW), track fragmentations (Frag.), mostly tracked (MT), and partially tracked (PT). The MOTA score is calculated by \text {MOTA} = 1 - \frac {\sum _t (\text {FN}_t + \text {FP}_t + \text {IDSW}_t)}{\sum _t \text {GT}_t}, where t is the frame index, GT is the number of ground-truth objects, FN and FP refer to false negative and false positive detection. The value of MOTA is in the range (−∞, 100]. It can be deemed as the combination of detection and tracking performance, and is widely used as the main metric for accessing multiple object tracking quality. MOTP is the average IoU value on all ground-truth bounding boxes and its assigned prediction. It describes the localized precision.

The rest of these metrics all reflect the quality of predicted tracklets. For detailed definitions and calculations of MOT metrics, please refer to [18].

We attach a decoding algorithm for multiple object tracking. The tracking algorithm mainly follows [42] which associates objects from successive frames purely based on the cost of Euclidean distance. The position of an object in the previous frame is complemented with a predictive positional tracking offset dˆ to infer its potential position in the next frame. Then, objects in previous and current frames are associated and propagate the object’s ID in a bipartite graph with a greedy algorithm based on the distance between their center 2D positions. Empirically, we do not further extend a tracklet if it cannot find a matched candidate.

## C. Ablation Study

We add an experiment on split train good weather in

Fig. 6 to analyze the change of the number of the selective feature vectors for temporal relational layers, where we vary the value K from 2 to 20. The detection performance consistently improved before K reached 8, but drop when continually increase the value of K. The scenario

Algorithm 1 Multiple Object Tracking Decoding

Require: Tt−1 = {(c, id)t−1 j }Mj=1: tracked objects in the previous frame t−1; Bˆt = {(ˆc, v, dˆ)ti}Ni=1 heatmap predictions of object centers ˆc, confidence v, and tracking offsets dˆ. Bˆt are sorted in a descending order according to v. Distance threshold k. Birth threshold b. 1: S ← ∅, Tt ← ∅ 2: W ← Cost(Bˆt , Tt−1) \triangleright Wij = ||ˆcti − dˆti, ˆct−1 j ||2 3: for i ← 1, N do 4: j ← arg minj /∈S Wij 5: if wij ≤ k then 6: Tt ← Tt ∪ (ˆcti, idt−1 j ) \triangleright Propagate matched id 7: S ← S ∪ {j} \triangleright Mark candidate j as tracked 8: else if vi ≥ b then 9: Tt ← Tt ∪ (ˆcti, New id) \triangleright Create a new track 10: end if 11: end for 12: return Tt indicates involving redundant objects in relation modeling could slightly corrupt the temporal relation learning. The value of K should be selected based on the average number of objects per frame but not including excessive noise. We empirically set K to 8 in our experiments.

## D. Additional Visualization Result

We present additional visualization results in Fig. 7 on object detection. In the detection, green bounding boxes are ground-truth annotations, while red are predictions. The same observations are confirmed in the additional visualizations. False positive predictions are mainly due to the ‘ghost’ objects in radar signals, and the rest are localized in the surroundings or outer space where the angular resolution is low.

## E. A Short Review of Radar Dataset

Besides the algorithmic design, many radar datasets are emerging which are crucial for machine learning research.

Among these datasets, radar data are currently presented in various data formats, i.e. radio frequency heatmap, radar reflection image, or point cloud. RadarScenes dataset [24] provide abundant point-wise annotations with doppler for automotive radar. However, there is no bounding box annotation for objects. Carrada dataset [19] records the rangeangle and range-Doppler heatmap. Their data are mainly recorded in experimental sites like parking lots but not in real driving environment. CRUW dataset [31] offers radar’s radio frequency images with camera-projected annotations. nuScenes [5] contains multi-modal data including Lidar, camera, and radar. However, radar data in nuScenes only afford sparse point cloud, while the Lidar and camera data are the

Figure 5. The backbone networks are inserted with several skip connections to collect features at different scales for predictions. Features selected for temporal relations modeling are attached with positional encoding to reveal the locality of objects.

Figure 6. Detection performance with varying K value. main advantage of this dataset. MulRan [12] and Oxford [2] datasets present high-resolution radar images for urban driving scenarios but without object-level annotation. In our paper, we conduct detection and tracking experiments on point cloud-based radar images in adverse weather from Radiate dataset [25], and every significant object has bounding box and tracking ID annotations for training.

Figure 7. Visualizations on object detection. From left to right and top to bottom, the figures are from: motorway-2-1, tiny foggy, junction-1-10, and fog-6-0. Green bounding boxes are ground-truth annotations, while red are model predictions from ‘Ours-ResNet18-w.
TRL.’
