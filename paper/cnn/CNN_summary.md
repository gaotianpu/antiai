# 卷积网络总结

## 一、卷积
https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d
https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Conv2D_cn.html#conv2d

paddle.nn.Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', weight_attr=None, bias_attr=None, data_format='NCHW')

torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

相同部分：
* in_channels, (int) - 输入图像的通道数。
* out_channels, (int) - 由卷积操作产生的输出的通道数。
* kernel_size,  (int|list|tuple) - 卷积核大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核的高和宽。如果为单个整数，表示卷积核的高和宽都等于该整数。
* stride=1, (int|list|tuple，可选) - 步长大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积沿着高和宽的步长。如果为单个整数，表示沿着高和宽的步长都等于该整数。默认值：1。
* padding=0, 
* dilation=1,  (int|list|tuple，可选) - 空洞大小。可以为单个整数或包含两个整数的元组或列表，分别表示卷积核中的元素沿着高和宽的空洞。如果为单个整数，表示高和宽的空洞都等于该整数。默认值：1。
* groups=1, (str，可选)：填充模式。包括 'zeros', 'reflect', 'replicate' 或者 'circular'。默认值：'zeros'
* padding_mode='zeros', (str，可选)：填充模式。包括 'zeros', 'reflect', 'replicate' 或者 'circular'。默认值：'zeros'

不同部分：
* weight_attr=None,  (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性
* bias_attr=None, ParamAttr|bool，可选)- 指定偏置参数属性的对象。若 bias_attr 为 bool 类型，只支持为 False，表示没有偏置参数。默认值为 None，表示使用默认的偏置参数属性。
* data_format='NCHW' , (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCHW"


参数量、计算量的公式

kernel_size, 正方形，长方形; 
stride， 下采样，替代pooling; 
padding,padding_mode

dilation, 空洞卷积
groups， 分组卷积

## 二、归一化
批归一化，
https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/BatchNorm_cn.html#batchnorm
https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html?highlight=batchnorm#torch.nn.BatchNorm2d


组归一化

## 三、激活函数
conv+batchnormal+activte


## 四、堆叠方式
1. pytorch
https://github.com/pytorch/vision/tree/main/torchvision/models
https://github.com/pytorch/examples/tree/main/mnist
https://github.com/pytorch/examples/blob/main/imagenet/

vgg，简单堆叠
inception, 旁路concat
resnet，残差连接
mobilenet等性能考虑的？
csp, 跨阶段部分连接

2. paddlepaddle


## 五、目标检测
输入尺寸不固定
路径聚合

## 六、实例分割

## 七、语义分割

## 应用
1. 智能相册 
* 适用于pc,移动设备,大屏幕，特化设备
* 图片导入，按时间分割组织文件夹？ 使用sqlite数据库存储meta信息
* 基本信息提取 (c++ 轻量图形lib)
* 去重(传统的，大规模快速去重算法) https://www.cnblogs.com/luxiaoxun/p/14392375.html
* 颜色区间，根据颜色条 检索图片？
* 基于cnn结构的去重，ab两张图片的关系，1.无交集，2.完全相同 3.a是从b剪切出来的，存在交集
* 人脸识别
* 物体识别
* 风格迁移，人物的漫画风格？
* 高清修复
* 检索，语音检索？
* 打标签，生成复杂的文本描述？
* 聚类

2. 导航、路径规划、3D场景构建

3. 无人机(卫星)俯瞰视角，目标检测、语义分割，数量，面积测算; 

