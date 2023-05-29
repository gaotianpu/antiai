from torchvision.io import read_image,ImageReadMode
import torchvision.transforms as T

# 加载图像，返回uint8的张量 C*H*W, 每个像素值在 [0,255] 区间
# 灰度图像:ImageReadMode.GRAY，RGB图像: .RGB，带透明度的RGB：.RGB_ALPHA
image = read_image("./flower.png", ImageReadMode.RGB) 
print(image.shape) 

# 将tensor转换 PIL Image，否则下面的ToTensor()操作会出错误提示：
# pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>
image = T.ToPILImage()(image)

# 将每个像素的值都除以255, 做max-min归一化处理， 得到一个[0,1]之间的。
# 没有这步，会出错误：Input tensor should be a float tensor. Got torch.uint8
image = T.ToTensor()(image)
print(image.shape)

# 标准化： output = (input - mean) / std
# 把值区间从 [0,1] 转为 [-1,1]区间 。 和网络里的激活函数有关
image = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
print(image.shape)