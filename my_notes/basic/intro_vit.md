# 视觉Transfomer 

Transformer在NLP领域的成功经验，正在被研究人员移植到机器视觉领域。

https://github.com/lucidrains/vit-pytorch/

## 1. ViT
结合以下3篇论文看
* ViT:[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](../paper/vit/ViT.md)
* SimpleViT: [Better plain ViT baselines for ImageNet-1k](../paper/vit/simpleViT.md)
* [Early Convolutions Help Transformers See Better](https://arxiv.org/abs/2106.14881) 

![Vit](../paper/images/vit/fig_1.png)<br/>
图1：ViT架构示意图。

综合上面的3篇论文，主要流程：
1. 根据设定的每个分块的宽高尺寸，将完整图像分成若干个分块；
2. 将所有图像分块过投影层，映射成像NLP的token概念。这个映射机制可以是最简单的单层线性投影，也可以是MLP，也可以是多层卷积网络；原始的输入：[batch,channel,height,width], 转换后[batch,patches,emb_dim].
3. 将映射得到的分块向量，和位置嵌入相加，位置嵌入选择和NLP一样，sincos, 可学习的(外推效果差，主流的都不在推荐)等等，采用2D和1D的位置嵌入效果差别不大，1D的也能表达二维图像的顺序关系；
4. 头部附加的[CLS] token 非必须，用全局平均池化能够达成更好打的效果，SimpleViT证明了这点；
5. 后续操作是标准的transfomer blocks , 无需赘述；

训练技巧：
1. 使用RandAug + MixUp 图像增广方式 (效果明显)
2. Batch-size，从4096改为1024； (效果明显)


重点关注下，输入图像转成transformer能处理的token这部分的逻辑：

```python
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
num_patches = (image_height // patch_height) * (image_width // patch_width)
patch_dim = channels * patch_height * patch_width

# RGB三个通道的颜色值，过线性再打平成n-d的向量
self.to_patch_embedding = nn.Sequential(
    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
    nn.LayerNorm(patch_dim),
    nn.Linear(patch_dim, dim),
    nn.LayerNorm(dim),
)

x = self.to_patch_embedding(img)
b, n, _ = x.shape

```

```python
# 另一种图像分块、投影方式
# 源：https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
if conv_stem_configs is not None:
    # As per https://arxiv.org/abs/2106.14881
    seq_proj = nn.Sequential()
    prev_channels = 3 #RGB
    for i, conv_stem_layer_config in enumerate(conv_stem_configs):
        seq_proj.add_module(
            f"conv_bn_relu_{i}",
            Conv2dNormActivation(
                in_channels=prev_channels,
                out_channels=conv_stem_layer_config.out_channels,
                kernel_size=conv_stem_layer_config.kernel_size,
                stride=conv_stem_layer_config.stride,
                norm_layer=conv_stem_layer_config.norm_layer,
                activation_layer=conv_stem_layer_config.activation_layer,
            ),
        )
        prev_channels = conv_stem_layer_config.out_channels
    seq_proj.add_module(
        "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
    )
    self.conv_proj: nn.Module = seq_proj
else: #简单点的投影方案
    self.conv_proj = nn.Conv2d(
        in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
    )

def _process_input(self, x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    p = self.patch_size
    n_h = h // p
    n_w = w // p

    # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
    x = self.conv_proj(x)
    # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    x = x.reshape(n, self.hidden_dim, n_h * n_w)

    # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
    # The self attention layer expects inputs in the format (N, S, E)
    # where S is the source sequence length, N is the batch size, E is the
    # embedding dimension
    x = x.permute(0, 2, 1)

    return x

```


## 2. MAE 
相关论文：
* [Masked Autoencoders Are Scalable Vision Learners](../paper/vit/MAE.md)
* [Position Prediction as an Effective Pretraining Strategy](https://arxiv.org/abs/2207.07611)

![MAE](../paper/images/mae/fig_1.png)<br/>
图2：MAE架构示意图。在预训练期间，随机掩码大部分(例如，75%)的分块子集。编码器应用于可见分块的小子集。掩码令牌在编码器之后引入，全套编码分块和掩码令牌由一个小解码器处理，该解码器以像素重建原始图像。在预训练之后，解码器被丢弃，编码器被应用于未损坏的图像(完整的分块集)以进行识别任务。

重点：
1. 基于图像的冗余度较大的前提下，掩码掉75%的部分。
2. 只对未掩码部分做encoder，但位置嵌入保留了，
3. [mask] token是没有经过encoder的，因此，它的值只能是在decoder中学习？
4. decoder只预测掩码部分的token。

推理阶段：
1. 图像不再掩码，所有图像分块都过encoder得到完整的向量表示。

```python
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mae.py
# 这版本的实现有缺陷，只考虑的训练，没考虑推理。
class MAE(nn.Module):
    def forward(self, img):
        device = img.device

        # get patches
        # encoder.to_patch_embedding[0]
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        # nn.Sequential(*encoder.to_patch_embedding[1:])
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None]
        # 未被掩码的，patch位置变化？
        tokens = tokens[batch_range, unmasked_indices]

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        # self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        # self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder    
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        # self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        # self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        pred_pixel_values = self.to_pixels(mask_tokens)

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss

```

```python
# https://github.com/facebookresearch/mae/blob/main/models_mae.py

```

## 3. FLIP: 图文理解
相关论文：
* CLIP [Learning Transferable Visual Models From Natural Language Supervision](../paper/Multimodal/CLIP.md)
* OpenCLIP [Reproducible scaling laws for contrastive language-image learning](../paper/Multimodal/OpenCLIP.md)
* FLIP [Scaling Language-Image Pre-training via Masking](../paper/Multimodal/FLIP.md)

![FLIP](../paper/images/FLIP/fig_2.png)</br>
图3：FLIP架构。在CLIP之后，我们对成对的图像和文本样本进行对比学习。我们随机掩码掉具有高掩码率的图像分块，并仅对可见分块进行编码。我们不执行掩码图像内容的重建。

重点：
1. 对比学习框架
2. 图片编码采用MAE方式，

拿MAE预训练出来的模型，图像单模态的语义表征，去FLIP再微调？


## 4. ViTDet 
基于ViT的目标检测、语义分割、实例分割
* [Benchmarking Detection Transfer Learning with Vision Transformers](../paper/vit/ViTDet_Benchmarking.md)
* [Exploring Plain Vision Transformer Backbones for Object Detection](../paper/vit/ViTDet.md)

![ViTDet](../paper/images/ViTDet/fig_1.png)<br/>
图4：典型的分层主干检测器(左)与普通主干检测器(右)。传统的分层主干网可以自然地适用于多尺度检测，例如，使用FPN。相反，我们探索从一个普通主干的最后一个大跨度(16)特征图构建一个简单的金字塔。


## 5. [Segment Anything](../paper/Multimodal/Segment_Anything.md)
![Segment Anything](../paper/images/SAM/fig_1.png)<br/>
图5：我们旨在通过引入三个相互关联的组件来构建分割的基础模型：提示分割任务、支持数据标注并通过提示工程将零样本迁移到一系列任务的分割模型（SAM），以及用于收集SA-1B的数据引擎，SA-1B是我们拥有超过10亿个掩码的数据集。