# Robust Speech Recognition via Large-Scale Weak Supervision
通过大规模弱监督进行稳健语音识别 2022.9.21 https://openai.com/research/whisper

## 阅读笔记
* 预训练目标、数据、架构
* 数据, 音频中的语音数据抽取？

## Abstract
We study the capabilities of speech processing systems trained simply to predict large amounts of transcripts of audio on the internet. When scaled to 680,000 hours of multilingual and multitask supervision, the resulting models generalize well to standard benchmarks and are often competitive with prior fully supervised results but in a zeroshot transfer setting without the need for any finetuning. When compared to humans, the models approach their accuracy and robustness. We are releasing models and inference code to serve as a foundation for further work on robust speech processing.

我们研究了经过简单训练以预测互联网上大量音频转录本的语音处理系统的能力。 当扩展到 680,000 小时的多语言和多任务监督时, 生成的模型可以很好地泛化到标准基准, 通常与之前的完全监督结果更具竞争力, 并且在 zeroshot 迁移设置中不需要任何微调。 与人类相比, 这些模型接近其准确性和稳健性。 我们正在发布模型和推理代码, 作为进一步研究稳健语音处理的基础。<!-- 预测 音频转录本 -->

## 1. Introduction
Progress in speech recognition has been energized by the development of unsupervised pre-training techniques exemplified by Wav2Vec 2.0 (Baevski et al., 2020). Since these methods learn directly from raw audio without the need for human labels, they can productively use large datasets of unlabeled speech and have been quickly scaled up to 1,000,000 hours of training data (Zhang et al., 2021), far more than the 1,000 or so hours typical of an academic supervised dataset. When fine-tuned on standard benchmarks, this approach has improved the state of the art, especially in a low-data setting.

以Wav2Vec 2.0为例的无监督预训练技术的发展推动了语音识别的进步(Baevski et al., 2020)。 由于这些方法直接从原始音频中学习而不需要人工标注, 因此它们可以高效地使用未标注语音的大型数据集, 并已迅速扩展到 1,000,000 小时的训练数据(Zhang et al., 2021), 远远超过 1,000小时 学术监督数据集的典型小时数左右。 当在标准基准上进行微调时, 这种方法改进了现有技术, 特别是在低数据设置中。<!-- 语音的无监督预训练？ -->

These pre-trained audio encoders learn high-quality representations of speech, but because they are purely unsupervised they lack an equivalently performant decoder mapping those representations to usable outputs, necessitating a finetuning stage in order to actually perform a task such as speech recognition(1 Baevski et al. (2021) is an exciting exception - having developed a fully unsupervised speech recognition system). This unfortunately limits their usefulness and impact as fine-tuning can still be a complex process requiring a skilled practitioner. There is an additional risk with requiring fine-tuning. Machine learning methods are exceedingly adept at finding patterns within a training dataset which boost performance on held-out data from the same dataset. However, some of these patterns are brittle and spurious and don’t generalize to other datasets and distributions. In a particularly disturbing example, Radford et al. (2021) documented a 9.2% increase in object classification accuracy when fine-tuning a computer vision model on the ImageNet dataset (Russakovsky et al., 2015) without observing any improvement in average accuracy when classifying the same objects on seven other natural image datasets. A model that achieves “superhuman” performance when trained on a dataset can still make many basic errors when evaluated on another, possibly precisely because it is exploiting those dataset-specific quirks that humans are oblivious to (Geirhos et al., 2020).

这些预训练的音频编码器学习高质量的语音表示, 但由于它们完全不受监督, 因此它们缺乏将这些表示映射到可用输出的等效性能解码器, 需要一个微调阶段才能实际执行语音识别等任务(1 Baevski et al., (2021) 是一个令人兴奋的例外 —— 开发了一个完全无监督的语音识别系统)。 不幸的是, 这限制了它们的实用性和影响, 因为微调仍然是一个复杂的过程, 需要熟练的从业者。 需要微调还有一个额外的风险。 机器学习方法非常擅长在训练数据集中寻找模式, 从而提高同一数据集中保留数据的性能。 然而, 其中一些模式是脆弱的和虚假的, 不会推广到其他数据集和分布。 在一个特别令人不安的例子中, Radford et al. (2021) 记录了在对 ImageNet 数据集(Russakovsky et al., 2015)上的计算机视觉模型进行微调时, 对象分类准确度提高了9.2%, 而在其他七个自然图像数据集上对相同对象进行分类时, 平均准确度没有任何提高. 在一个数据集上训练时达到“超人”性能的模型在另一个数据集上进行评估时仍然会犯许多基本错误, 这可能恰恰是因为它利用了人类忽略的那些特定于数据集的怪癖(Geirhos et al., 2020)。

This suggests that while unsupervised pre-training has improved the quality of audio encoders dramatically, the lack of an equivalently high-quality pre-trained decoder, combined with a recommended protocol of dataset-specific finetuning, is a crucial weakness which limits their usefulness and robustness. The goal of a speech recognition system should be to work reliably “out of the box” in a broad range of environments without requiring supervised fine-tuning of a decoder for every deployment distribution.

这表明虽然无监督预训练显著提高了音频编码器的质量, 但缺乏同等质量的预训练解码器, 再加上推荐的数据集特定微调协议, 是一个关键的弱点, 限制了它们的实用性和稳健性。 语音识别系统的目标应该是在广泛的环境中“开箱即用”地可靠工作, 而不需要为每个部署分布对解码器进行监督微调。<!--预训练解码器-->

As demonstrated by Narayanan et al. (2018), Likhomanenko et al. (2020), and Chan et al. (2021) speech recognition systems that are pre-trained in a supervised fashion across many datasets/domains exhibit higher robustness and generalize much more effectively to held-out datasets than models trained on a single source. These works achieve this by combining as many existing high-quality speech recognition datasets as possible. However, there is still only a moderate amount of this data easily available. SpeechStew (Chan et al., 2021) mixes together 7 pre-existing datasets totalling 5,140 hours of supervision. While not insignificant, this is still tiny compared to the previously mentioned 1,000,000 hours of unlabeled speech data utilized in Zhang et al. (2021).

正如 Narayanan et al. (2018), Likhomanenko et al. (2020), and Chan et al. (2021) 所证明的那样, 与在单一来源上训练的模型相比, 在许多数据集/领域中以监督方式预训练的语音识别系统表现出更高的稳健性, 并且更有效地泛化到保留的数据集。 这些工作通过尽可能多地结合现有的高质量语音识别数据集来实现这一目标。 然而, 仍然只有中等数量的数据很容易获得。 SpeechStew(Chan et al., 2021)将 7 个预先存在的数据集混合在一起, 总计 5,140 小时的监督。 虽然不是微不足道, 但与前面提到的 Zhang et al. (2021)使用的 1,000,000 小时未标注语音数据相比, 这仍然很小。

Recognizing the limiting size of existing high-quality supervised datasets, recent efforts have created larger datasets for speech recognition. By relaxing the requirement of goldstandard human-validated transcripts, Chen et al. (2021) and Galvez et al. (2021) make use of sophisticated automated  pipelines to scale weakly supervised speech recognition to 10,000 and 30,000 hours of noisier training data. This trade-off between quality and quantity is often the right call. Although understudied so far for speech recognition, recent work in computer vision has demonstrated that moving beyond gold-standard crowdsourced datasets such as ImageNet (Russakovsky et al., 2015) to much larger but weakly supervised datasets significantly improves the robustness and generalization of models (Mahajan et al., 2018; Kolesnikov et al., 2020).

认识到现有高质量监督数据集的大小有限, 最近的努力已经为语音识别创建了更大的数据集。 通过放宽对黄金标准人类验证转录本的要求, Chen et al. (2021) and Galvez et al. (2021) 通过大规模弱监督管道利用复杂的自动稳健语音识别将弱监督语音识别扩展到 10,000 和 30,000 小时的嘈杂训练数据。 这种质量和数量之间的权衡通常是正确的选择。 尽管到目前为止语音识别的研究不足, 但最近在计算机视觉方面的工作表明, 从 ImageNet(Russakovsky et al., 2015)等黄金标准众包数据集迁移到更大但监督较弱的数据集可以显著提高模型的稳健性和泛化性( Mahajan et al., 2018 ;Kolesnikov et al., 2020)。<!-- 通过大规模弱监督管道利用复杂的自动稳健语音识别 -->

Yet these new datasets are only a few times larger than the sum of existing high-quality datasets and still much smaller than prior unsupervised work. In this work we close that gap, scaling weakly supervised speech recognition the next order of magnitude to 680,000 hours of labeled audio data. We call our approach Whisper (2 If an acronym or basis for the name is desired, WSPSR standing for Web-scale Supervised Pretraining for Speech Recognition can be used). We demonstrate models trained at this scale transfer well to existing datasets zeroshot, removing the need for any dataset-specific fine-tuning to achieve high-quality results.

然而, 这些新数据集仅比现有高质量数据集的总和大几倍, 并且仍然比以前的无监督工作小得多。 在这项工作中, 我们缩小了这一差距, 将弱监督语音识别扩展到下一个数量级, 达到 680,000 小时的标注音频数据。 我们称我们的方法为 Whisper(如果需要该名称的首字母缩略词或基础，可以使用WSPSR，代表网络规模的语音识别监督预训练)。 我们展示了以这种规模训练的模型可以很好地迁移到现有数据集 zeroshot, 无需任何特定于数据集的微调即可获得高质量的结果。

In addition to scale, our work also focuses on broadening the scope of weakly supervised pre-training beyond English-only speech recognition to be both multilingual and multitask. Of those 680,000 hours of audio, 117,000 hours cover 96 other languages. The dataset also includes 125,000 hours of X→en translation data. We find that for sufficiently large models there is no drawback and even benefits to joint multilingual and multitask training.

除了规模之外, 我们的工作还侧重于扩大弱监督预训练的范围, 超越纯英语语音识别, 使其成为多语言和多任务的。 在这 680,000 小时的音频中, 117,000 小时涵盖了 96 种其他语言。 该数据集还包括 125,000 小时的 X→en 翻译数据。 我们发现, 对于足够大的模型, 联合多语言和多任务训练没有缺点, 甚至有好处。

Our work suggests that simple scaling of weakly supervised pre-training has been underappreciated so far for speech recognition. We achieve these results without the need for the self-supervision or self-training techniques that have been a mainstay of recent large-scale speech recognition work. To serve as a foundation for further research on robust speech recognition, we release inference code and models at the following URL: https://github.com/openai/whisper.

我们的工作表明, 到目前为止, 对于语音识别, 弱监督预训练的简单缩放一直未得到充分重视。 我们在不需要自监督或自我训练技术的情况下取得了这些结果, 这些技术已成为近期大规模语音识别工作的支柱。 为了作为进一步研究稳健语音识别的基础, 我们在以下网址发布了推理代码和模型：https://github.com/openai/whisper。

## 2. Approach
### 2.1. Data Processing
Following the trend of recent work leveraging web-scale text from the internet for training machine learning systems, we take a minimalist approach to data pre-processing. In contrast to a lot of work on speech recognition, we train Whisper models to predict the raw text of transcripts without any significant standardization, relying on the expressiveness of sequence-to-sequence models to learn to map between utterances and their transcribed form. This simplifies the speech recognition pipeline since it removes the need for a separate inverse text normalization step in order to produce naturalistic transcriptions.

随着最近利用互联网上的网络规模的文本来训练机器学习系统的趋势, 我们采用了一种极简的数据预处理方法。与语音识别方面的大量工作不同, 我们训练Whisper模型来预测转录本的原始文本, 而无需任何显著的标准化, 这依赖于序列到序列模型的表达能力, 以学习如何在话语与其转录形式之间进行映射。这简化了语音识别管道, 因为它不需要单独的反向文本规范化步骤来生成自然转录。<!-- 预测转录本的原始文本, 反向文本规范化 -->

We construct the dataset from audio that is paired with transcripts on the Internet. This results in a very diverse dataset covering a broad distribution of audio from many different environments, recording setups, speakers, and languages. While diversity in audio quality can help train a model to be robust, diversity in transcript quality is not similarly beneficial. Initial inspection showed a large amount of subpar transcripts in the raw dataset. To address this, we developed several automated filtering methods to improve transcript quality.

我们从与互联网上的转录本配对的音频构建数据集。这导致了一个非常多样化的数据集, 涵盖了来自许多不同环境、录音设置、扬声器和语言的广泛音频分布。虽然音频质量的多样性有助于训练模型的稳健性, 但转录质量的多样也没有同样的好处。初步检查显示原始数据集中有大量亚标准转录本。为了解决这个问题, 我们开发了几种自动过滤方法来提高转录质量。

Many transcripts on the internet are not actually humangenerated but the output of existing ASR systems. Recent research has shown that training on datasets of mixed human and machine-generated data can significantly impair the performance of translation systems (Ghorbani et al., 2021). In order to avoid learning “transcript-ese”, we developed many heuristics to detect and remove machine-generated transcripts from the training dataset. Many existing ASR systems output only a limited subset of written language which removes or normalizes away aspects that are difficult to predict from only audio signals such as complex punctuation (exclamation points, commas, and question marks), formatting whitespace such as paragraphs, or stylistic aspects such as capitalization. An all-uppercase or all-lowercase transcript is very unlikely to be human generated. While many ASR systems include some level of inverse text normalization, it is often simple or rule-based and still detectable from other unhandled aspects such as never including commas.

互联网上的许多转录本实际上不是人工生成的, 而是现有ASR系统的输出。最近的研究表明, 对混合人工和机器生成数据的数据集进行训练会严重影响翻译系统的性能(Ghorbani et al., 2021)。为了避免学习“转录本”, 我们开发了许多启发式方法来检测并从训练数据集中删除机器生成的转录本。许多现有的ASR系统仅输出有限的书面语言子集, 该子集仅从复杂的标点符号(感叹号、逗号和问号)、格式化空格(如段落)或风格方面(如大写)等音频信号中去除或规范化难以预测的方面。全大写或全小写的转录本不太可能是人类生成的。尽管许多ASR系统包括某种程度的反向文本规范化, 但它通常是简单的或基于规则的, 并且仍然可以从其他未处理的方面(例如从不包含逗号)检测到。
<!--启发式方法  删除机器生成的转录本 ; 人工的转录本从那里获得的？-->

We also use an audio language detector, which was created by fine-tuning a prototype model trained on a prototype version of the dataset on VoxLingua107 (Valk & Alum¨ae, 2021) to ensure that the spoken language matches the language of the transcript according to CLD2. If the two do not match, we don’t include the (audio, transcript) pair as a speech recognition training example in the dataset. We make an exception if the transcript language is English and add these pairs to the dataset as X→en speech translation training examples instead. We use fuzzy de-duping of transcript texts to reduce the amount of duplication and automatically generated content in the training dataset.

我们还使用了一个音频语言检测器, 该检测器是通过微调根据VoxLingua107(Valk&Alum¨ae, 2021)上的数据集原型版本训练的原型模型而创建的, 以确保口语符合CLD2中的抄本语言。如果两者不匹配, 我们就不将(音频、转录本)对作为语音识别训练样本包含在数据集中。如果转录本语言为英语, 我们将例外, 并将这些对作为X→en添加到数据集而不是语音翻译训练样本。我们使用转录文本的模糊去重复以减少重复量, 并在训练数据集中自动生成内容。

We break audio files into 30-second segments paired with the subset of the transcript that occurs within that time segment. We train on all audio, including segments where there is no speech (though with sub-sampled probability) and use these segments as training data for voice activity detection.

我们将音频文件分成30秒的片段, 与该时间段内出现的转录子集配对。我们对所有音频进行训练, 包括没有语音的片段(尽管具有亚采样概率), 并将这些片段用作语音活动检测的训练数据。

For an additional filtering pass, after training an initial model we aggregated information about its error rate on training data sources and performed manual inspection of these data sources sorting by a combination of both high error rate and data source size in order to identify and remove low-quality ones efficiently. This inspection showed a large amount of only partially transcribed or poorly aligned/misaligned transcripts as well as remaining low-quality machine-generated captions that filtering heuristics did not detect.

对于额外的过滤过程, 在训练初始模型之后, 我们在训练数据源上聚集关于其错误率的信息, 并通过高错误率和数据源大小的组合对这些数据源进行手动检查, 以便有效地识别和移除低质量的数据源。这项检查显示了大量仅部分转录或对齐不良/未对齐的转录本, 以及过滤启发法没有检测到的剩余低质量机器生成的字幕。

To avoid contamination, we perform de-duplication at a transcript level between the training dataset and the evaluation datasets we thought were at higher risk of overlap, namely TED-LIUM 3 (Hernandez et al., 2018).

为了避免污染, 我们在训练数据集和我们认为重叠风险较高的评估数据集(即TED-LIUM 3)之间执行转录水平的重复数据消除(Hernandez et al., 2018)。

### 2.2. Model
Since the focus of our work is on studying the capabilities of large-scale supervised pre-training for speech recognition, we use an off-the-shelf architecture to avoid confounding our findings with model improvements. We chose an encoder-decoder Transformer (Vaswani et al., 2017) as this architecture has been well validated to scale reliably. All audio is re-sampled to 16,000 Hz, and an 80-channel logmagnitude Mel spectrogram representation is computed on 25-millisecond windows with a stride of 10 milliseconds. For feature normalization, we globally scale the input to be between -1 and 1 with approximately zero mean across the pre-training dataset. The encoder processes this input representation with a small stem consisting of two convolution layers with a filter width of 3 and the GELU activation function (Hendrycks & Gimpel, 2016) where the second convolution layer has a stride of two. Sinusoidal position embeddings are then added to the output of the stem after which the encoder Transformer blocks are applied. The transformer uses pre-activation residual blocks (Child et al., 2019), and a final layer normalization is applied to the encoder output. The decoder uses learned position embeddings and tied input-output token representations (Press & Wolf, 2017). The encoder and decoder have the same width and number of transformer blocks. Figure 1 summarizes the model architecture.

由于我们的工作重点是研究语音识别的大规模监督预训练的能力, 我们使用现成的架构来避免将我们的发现与模型改进混淆。我们选择了编码器-解码器转换器(Vaswani et al., 2017), 因为该架构已被充分验证, 可以可靠地扩展。所有音频都被重新采样到16000 Hz, 并且在25毫秒的窗口上以10毫秒的步长计算80通道对数幅度Mel谱图表示。对于特征归一化, 我们在预训练数据集上全局缩放输入, 使其介于-1和1之间, 平均值近似为零。编码器使用由两个卷积层和GELU激活函数(Hendrycks&Gimpel, 2016)组成的小主干处理该输入表示, 其中第二个卷积层的步长为2。然后将正弦位置嵌入添加到阀杆的输出, 然后应用编码器Transformer块。Transformer使用预激活残差块(Child et al., 2019), 并将最终层归一化应用于编码器输出。解码器使用学习的位置嵌入和绑定的输入输出令牌表示(Press&Wolf, 2017)。编码器和解码器具有相同的Transformer块宽度和数量。图1总结了模型架构。

Figure 1. Overview of our approach. A sequence-to-sequence Transformer model is trained on many different speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. All of these tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing for a single model to replace many different stages of a traditional speech processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets, as further explained in Section 2.3.
图1。我们的方法概述。序列到序列变换器模型被训练用于许多不同的语音处理任务, 包括多语言语音识别、语音翻译、口语识别和语音活动检测。所有这些任务都被联合表示为要由解码器预测的令牌序列, 从而允许单个模型替换传统语音处理流水线的许多不同阶段。多任务训练格式使用一组特殊的令牌作为任务指定符或分类目标, 如第2.3节所述。


We use the same byte-level BPE text tokenizer used in GPT-2 (Sennrich et al., 2015; Radford et al., 2019) for the Englishonly models and refit the vocabulary (but keep the same size) for the multilingual models to avoid excessive fragmentation on other languages since the GPT-2 BPE vocabulary is English only.

我们将GPT-2中使用的相同字节级BPE文本令牌器(Sennrich et al., 2015; Radford et al., 2019)用于纯英语模型, 并重新调整多语言模型的词汇表(但保持相同大小), 以避免其他语言上的过度碎片化, 因为GPT-2 BPE词汇表仅为英语。

### 2.3. Multitask Format
Although predicting which words were spoken in a given audio snippet is a core part of the full speech recognition problem and extensively studied in research, it is not the only part. A fully featured speech recognition system can involve many additional components such as voice activity detection, speaker diarization, and inverse text normalization. These components are often handled separately, resulting in a relatively complex system around the core speech recognition model. To reduce this complexity, we would like to have a single model perform the entire speech processing pipeline, not just the core recognition part. An important consideration here is the interface for the model. There are many different tasks that can be performed on the same input audio signal: transcription, translation, voice activity detection, alignment, and language identification are some examples.

虽然预测给定音频片段中说出的单词是完整语音识别问题的核心部分, 并在研究中得到了广泛研究, 但这并不是唯一的部分。一个功能齐全的语音识别系统可以包括许多附加组件, 如语音活动检测、说话人日记和反向文本规范化。这些组件通常是单独处理的, 导致围绕核心语音识别模型的系统相对复杂。为了降低这种复杂性, 我们希望让一个模型来执行整个语音处理管道, 而不仅仅是核心识别部分。这里的一个重要考虑是模型的接口。在同一输入音频信号上可以执行许多不同的任务：转录、翻译、语音活动检测、对齐和语言识别是一些样本。

For this kind of one-to-many mapping to work with a single model, some form of task specification is necessary. We use a simple format to specify all tasks and conditioning information as a sequence of input tokens to the decoder. Since our decoder is an audio-conditional language model, we also train it to condition on the history of text of the transcript in the hope that it will learn to use longer-range text context to resolve ambiguous audio. Specifically, with some probability we add the transcript text preceding the current audio segment to the decoder’s context. We indicate the beginning of prediction with a \<|startoftranscript|> token.

为了使这种一对多映射与单个模型一起工作, 需要某种形式的任务规范。我们使用一种简单的格式将所有任务和条件信息指定为解码器的输入令牌序列。由于我们的解码器是一个音频条件语言模型, 我们还训练它以调节转录本的文本历史, 希望它能够学习使用更长范围的文本上下文来解决歧义音频。具体地说, 在某种概率下, 我们将当前音频片段之前的转录文本添加到解码器的上下文中。我们用\<|startofranscript|>令牌表示预测的开始。

First, we predict the language being spoken which is represented by a unique token for each language in our training set (99 total). These language targets are sourced from the aforementioned VoxLingua107 model. In the case where there is no speech in an audio segment, the model is trained to predict a \<|nospeech|> token indicating this. The next token specifies the task (either transcription or translation) with an \<|transcribe|> or \<|translate|> token. After this, we specify whether to predict timestamps or not by including a \<|notimestamps|> token for that case. At this point, the task and desired format is fully specified, and the output begins. For timestamp prediction, we predict time relative to the current audio segment, quantizing all times to the nearest 20 milliseconds which matches the native time resolution of Whisper models, and add additional tokens to our vocabulary for each of these. We interleave their prediction with the caption tokens: the start time token is predicted before each caption’s text, and the end time token is predicted after. When a final transcript segment is only partially included in the current 30- second audio chunk, we predict only its start time token for the segment when in timestamp mode, to indicate that the subsequent decoding should be performed on an audio window aligned with that time, otherwise we truncate the audio to not include the segment. Lastly, we add a \<|endoftranscript|> token. We only mask out the training loss over the previous context text, and train the model to predict all other tokens. Please see Figure 1 for an overview of our format and training setup.

首先, 我们预测所说的语言, 该语言由训练集中每种语言的唯一令牌表示(共99个)。这些语言目标源于上述VoxLingua107模型。在音频段中没有语音的情况下, 训练模型以预测表示这一点的令牌。下一个令牌使用\<|transcript|>或\<|translate|>令牌指定任务(转录或翻译)。之后, 我们通过在这种情况下包含\<|notimestamps|>令牌来指定是否预测时间戳。此时, 任务和所需格式已完全指定, 输出开始。对于时间戳预测, 我们预测与当前音频段相关的时间, 将所有时间量化为最接近的20毫秒, 这与Whisper模型的本地时间分辨率相匹配, 并为每个时间段向词汇表中添加额外的令牌。我们将它们的预测与字幕令牌交错：开始时间令牌在每个字幕文本之前预测, 结束时间令牌在之后预测。当最终转录片段仅部分包含在当前的30秒音频块中时, 我们仅预测该片段在时间戳模式下的开始时间令牌, 以指示后续解码应在与该时间对齐的音频窗口上执行, 否则我们截断音频以不包含该片段。最后, 我们添加一个\<|endotranscript|>令牌。我们只掩盖了先前上下文文本的训练损失, 并训练模型来预测所有其他令牌。请参见图1, 了解我们的格式和训练设置。


### 2.4. Training Details
We train a suite of models of various sizes in order to study the scaling properties of Whisper. Please see Table 1 for an overview. We train with data parallelism across accelerators using FP16 with dynamic loss scaling and activation checkpointing (Griewank & Walther, 2000; Chen et al., 2016). Models were trained with AdamW (Loshchilov & Hutter, 2017) and gradient norm clipping (Pascanu et al., 2013) with a linear learning rate decay to zero after a warmup over the first 2048 updates. A batch size of 256 segments was used, and the models are trained for 220 updates which is between two and three passes over the dataset. Due to only training for a few epochs, over-fitting is not a large concern, and we do not use any data augmentation or regularization and instead rely on the diversity contained within such a large dataset to encourage generalization and robustness. Please see Appendix F for full training hyperparameters.(3 After the original release of Whisper, we trained an additional Large model (denoted V2) for 2.5X more epochs while adding SpecAugment (Park et al., 2019), Stochastic Depth (Huang et al., 2016), and BPE Dropout (Provilkov et al., 2019) for regularization. Reported results have been updated to this improved model unless otherwise specified.)

为了研究Whisper的缩放属性, 我们训练了一组不同大小的模型。概述请参见表1。我们使用具有动态损失缩放和激活检查点的FP16跨加速器进行数据并行训练(Griewank&Walther, 2000; Chen et al., 2016)。使用AdamW(Loshchilov&Hutter, 2017)和梯度范数裁剪(Pascanu et al., 2013)训练模型, 在前2048次更新的预热后, 线性学习率衰减为零。使用了256个片段的批量大小, 并对模型进行了220次更新的训练, 这在数据集的两次到三次之间。由于只有几个时期的训练, 过度拟合不是一个大问题, 我们不使用任何数据增广或正则化, 而是依赖于如此大的数据集中包含的多样性来鼓励泛化和稳健性。有关完整的训练超参数, 请参见附录F。(3 在最初发布Whisper之后, 我们训练了一个额外的大型模型(表示为V2), 用于2.5倍多的时间段, 同时添加SpecAugment(Park et al., 2019)、随机深度(Huang et al., 2016)和BPE Dropout(Provilkov et al., 2019)进行正则化。除非另有规定, 报告结果已更新为该改进模型。)

Model | Layers | Width | Heads | Parameters
--- | --- | --- | --- | ---
Tiny|4|384|6|39M
Base|6|512|8|74M
Small|12|768|12|244M
Medium|24|1024|16|769M
Large|32|1280|20|1550M

Table 1. Architecture details of the Whisper model family. 
表1。Whisper模型系列的架构细节。

During early development and evaluation we observed that Whisper models had a tendency to transcribe plausible but almost always incorrect guesses for the names of speakers. This happens because many transcripts in the pre-training dataset include the name of the person who is speaking, encouraging the model to try to predict them, but this information is only rarely inferable from only the most recent 30 seconds of audio context. To avoid this, we fine-tune Whisper models briefly on the subset of transcripts that do not include speaker annotations which removes this behavior.

在早期开发和评估过程中, 我们观察到Whisper模型倾向于对说话者的名字进行看似合理但几乎总是不正确的猜测。这是因为训练前数据集中的许多记录都包含了说话的人的名字, 这鼓励模型尝试预测他们, 但仅从最近30秒的音频上下文中很少能推断出这些信息。为了避免这种情况, 我们对Whisper模型进行了简短的微调, 该模型针对不包含消除这种行为的说话者注释的转录子集。

## 3. Experiments
### 3.1. Zero-shot Evaluation
The goal of Whisper is to develop a single robust speech processing system that works reliably without the need for dataset specific fine-tuning to achieve high-quality results on specific distributions. To study this capability, we reuse a wide set of existing speech processing datasets to check whether Whisper is able to generalize well across domains, tasks, and languages. Instead of using the standard evaluation protocol for these datasets, which include both a train and test split, we evaluate Whisper in a zero-shot setting without using any of the training data for each of these datasets so that we are measuring broad generalization.

Whisper的目标是开发一个可靠工作的单一稳健语音处理系统, 而无需特定于数据集的微调, 从而在特定分布上获得高质量的结果。为了研究这种能力, 我们重用了大量现有的语音处理数据集, 以检查Whisper是否能够很好地跨领域、任务和语言进行概括。我们没有对这些数据集使用标准评估协议(包括训练和测试分割), 而是在零样本设置下评估Whisper, 而不使用这些数据集的任何训练数据, 因此我们正在测量广泛的通用性。

### 3.2. Evaluation Metrics
Speech recognition research typically evaluates and compares systems based on the word error rate (WER) metric. However, WER, which is based on string edit distance, penalizes all differences between the model’s output and the reference transcript including innocuous differences in transcript style. As a result, systems that output transcripts that would be judged as correct by humans can still have a large WER due to minor formatting differences. While this poses a problem for all transcribers, it is particularly acute for zero-shot models like Whisper, which do not observe any examples of specific datasets transcript formats.

语音识别研究通常基于单词错误率(WER)度量来评估和比较系统。然而, 基于字符串编辑距离的WER惩罚了模型输出和参考转录之间的所有差异, 包括转录样式的无害差异。因此, 输出人类判断为正确的转录本的系统, 由于格式上的细微差异, 仍然会有较大的WER。虽然这对所有转录者都是一个问题, 但对于Whisper这样的零样本模型来说, 这一问题尤为严重, 因为它们没有观察到任何特定数据集转录格式的例子。

This is not a novel observation; the development of evaluation metrics that better correlate with human judgement is an active area of research, and while there are some promising methods, none have seen widespread adoption for speech recognition yet. We opt to address this problem with extensive standardization of text before the WER calculation to minimize penalization of non-semantic differences. Our text normalizer was developed through iterative manual inspection to identify common patterns where naive WER penalized Whisper models for an innocuous difference. Appendix C includes full details. For several datasets, we observe WER drops of up to 50 percent usually due to a quirk such as a dataset’s reference transcripts seperating contractions from words with whitespace. We caution this development procedure comes at a risk of overfitting to the transcription style of Whisper models which we investigate in Section 4.4. We are releasing the code for our text normalizer to allow for easy comparison and to help others study the performance of speech recognition systems in out-of-distribution settings.

这不是一个新奇的观察; 开发更好地与人类判断相关的评估指标是一个活跃的研究领域, 尽管有一些有前途的方法, 但尚未有一种方法被广泛应用于语音识别。我们选择在WER计算之前通过文本的广泛标准化来解决这个问题, 以最小化非语义差异的惩罚。我们的文本规范化器是通过迭代手动检查来开发的, 以识别天真的WER惩罚Whisper模型产生无害差异的常见模式。附录C包括全部细节。对于几个数据集, 我们观察到WER下降了高达50%, 这通常是由于数据集的参考转录本将缩略词与带空格的单词分隔开等怪癖所致。我们警告说, 这种开发程序有可能过度适应我们在第4.4节中研究的Whisper模型的转录风格。我们正在发布文本规范化器的代码, 以便于比较, 并帮助其他人研究语音识别系统在非分布环境中的性能。

### 3.3. English Speech Recognition
In 2015, Deep Speech 2 (Amodei et al., 2015) reported a speech recognition system matched human-level performance when transcribing the LibriSpeech test-clean split. As part of their analysis they concluded: “Given this result, we suspect that there is little room for a generic speech system to further improve on clean read speech without further domain adaptation.” Yet seven years later the SOTA WER on LibriSpeech test-clean has dropped another 73% from their 5.3% to 1.4% (Zhang et al., 2021), far below their reported human-level error rate of 5.8%. Despite this massive and unanticipated further improvement in performance on held-out but in-distribution data, speech recognition models trained on LibriSpeech remain far above human error rates when used in other settings. What explains this gap between reportedly superhuman performance in-distribution and subhuman performance out-of-distribution?

2015年, 深度语音2(Amodei et al., 2015)报告了一种语音识别系统, 在转录LibriSpeech测试干净分割时, 该系统与人类水平的性能相匹配。作为分析的一部分, 他们得出结论：“考虑到这一结果, 我们怀疑通用语音系统在没有进一步的领域适应的情况下, 几乎没有进一步改进干净阅读语音的空间。”然而七年后, 自由语音测试清洁上的SOTA WER又下降了73%, 从5.3%降至1.4%(Zhang et al., 2021), 远远低于他们报告的5.8%的人为错误率。尽管在延迟但分布数据上的性能有了巨大的、意想不到的进一步提高, 但在LibriSpeech上训练的语音识别模型在其他环境中使用时仍然远远高于人为错误率。是什么解释了据报道在发行中的超人表现和发行中的非人表现之间的差距？

We suspect a large part of this gap between human and machine behavior is due to conflating different capabilities being measured by human and machine performance on a test set. This claim may seem confusing at first; if both humans and machines are taking the same test, how can it be that different skills are being tested? The difference arises not in the testing but in how they trained for it. Humans are often asked to perform a task given little to no supervision on the specific data distribution being studied. Thus human performance is a measure of out-of-distribution generalization. But machine learning models are usually evaluated after training on a large amount of supervision from the evaluation distribution, meaning that machine performance is instead a measure of in-distribution generalization. While both humans and machines are being evaluated on the same test data, two quite different abilities are being measured due to a difference in train data.

我们怀疑, 人和机器行为之间的差距很大一部分是由于将测试集上的人和机器性能所衡量的不同能力混为一谈。这一说法起初可能令人困惑; 如果人类和机器都在接受相同的测试, 那么不同的技能又怎么会被测试呢？不同之处不在于测试, 而在于他们是如何训练的。人类经常被要求执行一项任务, 而对所研究的特定数据分布几乎没有监督。因此, 人的表现是对分布外泛化的度量。但是, 机器学习模型通常在经过大量评估分布监督的训练后进行评估, 这意味着机器性能是分布内泛化的度量。虽然人类和机器都在同一测试数据上进行评估, 但由于训练数据的差异, 两种截然不同的能力正在被测量。

Whisper models, which are trained on a broad and diverse distribution of audio and evaluated in a zero-shot setting, could potentially match human behavior much better than existing systems. To study whether this is the case (or whether the difference between machine and human performance is due to yet-to-be-understood factors) we can compare Whisper models with both human performance and standard fine-tuned machine learning models and check which they more closely match.

Whisper模型在广泛多样的音频分布上进行了训练, 并在零样本设置下进行了评估, 可能比现有系统更符合人类行为。为了研究这种情况是否属实(或者机器和人的性能之间的差异是否是由于尚未理解的因素), 我们可以将Whisper模型与人的性能和标准的微调机器学习模型进行比较, 并检查它们更接近的匹配。

Figure 2. Zero-shot Whisper models close the gap to human robustness. Despite matching or outperforming a human on LibriSpeech dev-clean, supervised LibriSpeech models make roughly twice as many errors as a human on other datasets demonstrating their brittleness and lack of robustness. The estimated robustness frontier of zero-shot Whisper models, however, includes the 95% confidence interval for this particular human.

图2:零炮Whisper模型缩小了人类稳健性的差距。尽管在LibriSpeech开发环境中与人类相匹配或优于人类, 但受监督的LibriSpeech模型在其他数据集上产生的错误大约是人类的两倍, 表明其脆弱性和稳健性不足。然而, 零炮Whisper模型的估计稳健性边界包括该特定人类的95%置信区间。

To quantify this difference, we examine both overall robustness, that is average performance across many distributions/datasets, and effective robustness, introduced by Taori et al. (2020), which measures the difference in expected performance between a reference dataset, which is usually in-distribution, and one or more out-of-distribution datasets. A model with high effective robustness does better than expected on out-of-distribution datasets as a function of its performance on the reference dataset and approaches the ideal of equal performance on all datasets. For our analysis, we use LibriSpeech as the reference dataset due to its central role in modern speech recognition research and the availability of many released models trained on it, which allows for characterizing robustness behaviors. We use a suite of 12 other academic speech recognition datasets to study out-of-distribution behaviors. Full details about these datasets can be found in Appendix A.

为了量化这一差异, 我们检查了总体稳健性, 即许多分布/数据集的平均性能, 以及Tauri et al., (2020)引入的有效稳健性, 其测量了通常处于分布中的参考数据集与一个或多个分布外数据集之间的预期性能差异。作为其在参考数据集上的性能的函数, 具有高效稳健性的模型在非分布数据集上比预期的要好, 并且接近所有数据集上性能相同的理想。在我们的分析中, 我们使用LibriSpeech作为参考数据集, 这是因为它在现代语音识别研究中的核心作用, 以及在其上训练的许多发布模型的可用性, 这允许表征稳健性行为。我们使用一套其他12个学术语音识别数据集来研究非分布行为。有关这些数据集的详情, 请参见附录A。

Our main findings are summarized in Figure 2 and Table 2. Although the best zero-shot Whisper model has a relatively unremarkable LibriSpeech clean-test WER of 2.5, which is roughly the performance of modern supervised baseline or the mid-2019 state of the art, zero-shot Whisper models have very different robustness properties than supervised LibriSpeech models and out-perform all benchmarked LibriSpeech models by large amounts on other datasets. Even the smallest zero-shot Whisper model, which has only 39 million parameters and a 6.7 WER on LibriSpeech test-clean is roughly competitive with the best supervised LibriSpeech model when evaluated on other datasets. When compared to a human in Figure 2, the best zero-shot Whisper models roughly match their accuracy and robustness. For a detailed breakdown of this large improvement in robustness, Table 2 compares the performance of the best zero-shot Whisper model with a supervised LibriSpeech model that has the closest performance to it on LibriSpeech test-clean. Despite their very close performance on the reference distribution, the zero-shot Whisper model achieves an average relative error reduction of 55.2% when evaluated on other speech recognition datasets.

我们的主要发现汇总在图2和表2中。尽管最佳的零样本Whisper模型的LibriSpeech干净测试WER值为2.5, 相对来说并不显著, 这大致相当于现代监督基线或2019年年中的最新水平, 但零样本Whister模型的稳健性属性与监督的Libri语音模型大不相同, 并且在其他数据集上的表现远远超过所有基准的Libri言语模型。即使是最小的零炮Whisper模型, 它只有3900万个参数, 在LibriSpeech测试中的WER为6.7, 在其他数据集上进行评估时, 也与监管最好的LibriSpeech模型大致相当。与图2中的人类相比, 最佳零炮Whisper模型的精度和稳健性大致相当。为了详细分析这种在稳健性方面的巨大改进, 表2将最佳零炮Whisper模型的性能与受监督的LibriSpeech模型进行了比较, 该模型在LibriSpeechtest clean上具有最接近的性能。尽管零炮Whisper模型在参考分布上的性能非常接近, 但在其他语音识别数据集上进行评估时, 其平均相对误差降低了55.2%。

Table 2. Detailed comparison of effective robustness across various datasets. Although both models perform within 0.1% of each other on LibriSpeech, a zero-shot Whisper model performs much better on other datasets than expected for its LibriSpeech performance and makes 55.2% less errors on average. Results reported in word error rate (WER) for both models after applying our text normalizer. 
表2。对不同数据集的有效稳健性进行详细比较。尽管这两种模型在LibriSpeech上的性能相差0.1%, 但零炮Whisper模型在其他数据集上的性能比LibriSpeeh预期的要好得多, 平均误差减少55.2%。应用我们的文本归一化器后, 两个模型的结果均以单词错误率(WER)报告。

This finding suggests emphasizing zero-shot and out-ofdistribution evaluations of models, particularly when attempting to compare to human performance, to avoid overstating the capabilities of machine learning systems due to misleading comparisons.

这一发现表明, 应强调模型的零样本和分布外评估, 尤其是在试图与人类表现进行比较时, 以避免由于误导性比较而夸大机器学习系统的能力。

### 3.4. Multi-lingual Speech Recognition
In order to compare to prior work on multilingual speech recognition, we report results on two low-data benchmarks: Multilingual LibriSpeech (MLS) (Pratap et al., 2020b) and VoxPopuli (Wang et al., 2021) in Table 3.

为了与之前的多语言语音识别工作进行比较, 我们在表3中报告了两个低数据基准测试的结果：多语言LibriSpeech(MLS)(Pratap et al., 2020b)和VoxPopuli(Wang et al., 2021)。

Whisper performs well on Multilingual LibriSpeech, outperforming XLS-R (Babu et al., 2021), mSLAM (Bapna et al., 2022), and Maestro (Chen et al., 2022b) in a zero-shot setting. We caution that we do use a simple text standardizer for this result which prevents direct comparison or claims of SOTA performance. On VoxPopuli, however, Whisper significantly underperforms prior work and only beats the VP-10K+FT baseline from the original paper. We suspect the underperformance of Whisper models on VoxPopuli could be due to other models including this distribution as a major source for their unsupervised pre-training data and the dataset having significantly more supervised data, which benefits fine-tuning. While MLS has 10 hours of training data per language, the average amount of training data per language is roughly 10× higher for VoxPopuli.

Whisper在多语言LibriSpeech上表现良好, 在零样本设置下优于XLS-R(Babu et al., 2021)、mSLAM(Bapna et al., 2022)和Maestro(Chen et al., 2022b)。我们注意到, 我们确实使用了一个简单的文本标准化器来实现这一结果, 从而防止直接比较或声明SOTA性能。然而, 在VoxPopuli上, Whisper的表现明显低于之前的工作, 仅超过了原始论文的VP-10K+FT基线。我们怀疑VoxPopuli上Whisper模型的表现不佳可能是由于其他模型, 包括该分布作为其无监督预训练数据的主要来源, 以及数据集具有明显更多的监督数据, 这有利于微调。虽然MLS每种语言有10小时的训练数据, 但VoxPopuli每种语言的平均训练数据量大约高出10倍。

Figure 3. Correlation of pre-training supervision amount with downstream speech recognition performance. The amount of pre-training speech recognition data for a given language is very predictive of zero-shot performance on that language in Fleurs.
图3。训练前监督量与下游语音识别性能的相关性。给定语言的预训练语音识别数据量非常能预测Fleurs中该语言的零样本性能。

Table 3. Multilingual speech recognition performance. Zeroshot Whisper improves performance on Multilingual LibriSpeech (MLS) but is still significantly behind both Maestro, XLS-R, and mSLAM on VoxPopuli. 
表3。多语言语音识别性能。Zeroshot Whisper提高了多语言LibriSpeech(MLS)的性能, 但在VoxPopuli上仍然明显落后于Maestro、XLS-R和mSLAM。

These two benchmarks are somewhat narrow since they only include 15 unique languages, almost all of which are in the Indo-European language family and many of which are high-resource languages. These benchmarks only provide limited coverage and room to study Whisper models multilingual capabilities which include training data for speech recognition in 75 languages. To study the performance of Whisper more broadly we also report performance on the Fleurs dataset (Conneau et al., 2022). In particular, we were interested in studying the relationship between the amount of training data we have for a given language and the resulting downstream zero-shot performance for that language. We visualize this relation in Figure 3. We find a strong squared correlation coefficient of 0.83 between the log of the word error rate and the log of the amount of training data per language. Checking the regression coefficient for a linear fit to these log-log values results in an estimate that WER halves for every 16× increase in training data. We also observed that many of the largest outliers in terms of worse than expected performance according to this trend are languages that have unique scripts and are more distantly related to the Indo-European languages making up the majority of the training dataset such as Hebrew (HE), Telugu (TE), Chinese (ZH), and Korean (KO). These differences could be due to a lack of transfer due to linguistic distance, our byte level BPE tokenizer being a poor match for these languages, or variations in data quality.

这两个基准有点窄, 因为它们只包括15种独特的语言, 几乎所有语言都属于印欧语系, 其中许多是高资源语言。这些基准只提供了有限的覆盖范围和研究Whisper模型多语言能力的空间, 其中包括75种语言的语音识别训练数据。为了更广泛地研究Whisper的性能, 我们还报告了Fleurs数据集的性能(Conneau et al., 2022)。特别是, 我们有兴趣研究给定语言的训练数据量与该语言的下游零样本性能之间的关系。我们在图3中可视化了这种关系。我们发现, 单词错误率的对数和每种语言的训练数据量的对数之间的强平方相关系数为0.83。检查回归系数与这些对数对数值的线性拟合, 得出训练数据每增加16倍, WER就会减半的估计。我们还观察到, 根据这一趋势, 在表现比预期差的方面, 许多最大的异常值是具有独特脚本的语言, 并且与构成训练数据集的大多数的印欧语言(如希伯来语(HE)、泰卢固语(TE)、汉语(ZH)和韩语(KO))有着更遥远的关系。这些差异可能是由于语言距离、我们的字节级BPE令牌器与这些语言不匹配或数据质量变化导致的迁移不足。

Figure 4. Correlation of pre-training supervision amount with downstream translation performance. The amount of pretraining translation data for a given language is only moderately predictive of Whisper’s zero-shot performance on that language in Fleurs. 
图4。训练前监督金额与下游翻译绩效的相关性。给定语言的预训练翻译数据量只能适度预测Whisper在Fleurs中对该语言的零样本表现。

Table 4. X→en Speech translation performance. Zero-shot Whisper outperforms existing models on CoVoST2 in the overall, medium, and low resource settings but still moderately underperforms on high-resource languages compared to prior directly supervised work.
表4。十、→en语音翻译性能。零炮Whisper在总体、中等和低资源设置方面优于CoVoST2上的现有模型, 但与之前直接监督的工作相比, 在高资源语言上仍表现不佳。

Language ID | Fleurs 
--- | --- 
w2v-bert-51 (0.6B) | 71.4 
mSLAM-CTC (2B) | 77.7
Zero-shot Whisper | 64.5

Table 5. Language identification performance. Zero-shot Whisper’s accuracy at language identification is not competitive with prior supervised results on Fleurs. This is partially due to Whisper being heavily penalized for having no training data for 20 of Fleurs languages.
表5。语言识别性能。Zero-shot Whisper在语言识别方面的准确性与Fleurs之前的监督结果不相竞争。这部分是因为Whisper因为没有20种Fleurs语言的训练数据而受到严重处罚。

### 3.5. Translation
We study the translation capabilities of Whisper models by measuring their performance on the X→en subset of CoVoST2 (Wang et al., 2020b). We compare with Maestro, mSLAM, and XLS-R, the highest-performing prior work. We achieve a new state of the art of 29.1 BLEU zero-shot without using any of the CoVoST2 training data. We attribute this to the 68,000 hours of X→en translation data for these languages in our pre-training dataset which, although noisy, is vastly larger than the 861 hours of training data for X→en translation in CoVoST2. Since Whisper evaluation is zero-shot, it does particularly well on the lowest resource grouping of CoVoST2, improving over mSLAM by 6.7 BLEU. Conversely, the best Whisper model does not actually improve over Maestro and mSLAM on average for the highest resource languages.

我们通过测量Whisper模型在X→en上的性能来研究其翻译能力→CoVoST2的子集(Wang et al., 2020b)。我们与Maestro、mSLAM和XLS-R进行了比较, 它们是性能最高的前作。我们在不使用任何CoVoST2训练数据的情况下实现了29.1 BLEU零炮的新技术状态。我们将此归因于X的68000小时→在我们的预训练数据集中, 这些语言的翻译数据虽然有噪音, 但远远大于X→en的861小时训练数据→在CoVoST2中进行翻译。由于Whisper评估为零炮, 它在CoVoST2的最低资源分组上表现特别好, 比mSLAM提高了6.7 BLEU。相反, 对于资源最高的语言, 最好的Whisper模型实际上并没有比Maestro和mSLAM更好。

For an additional analysis on an even wider set of languages, we also re-purpose Fleurs, which is a speech recognition dataset, as a translation dataset. Since the same sentences are transcribed for every language we use the English transcripts as reference translations. In Figure 4 we visualize the correlation between the amount of translation training data per language and the resulting zero-shot BLEU score on Fleurs. While there is a clear trend of improvement with increasing training data, the squared correlation coefficient is much lower than the 0.83 observed for speech recognition and only 0.24. We suspect this is partly caused by the noisier training data due to errors in audio language identification. As an example, Welsh (CY) is an outlier with much worse than expected performance at only 13 BLEU despite supposedly having 9,000 hours of translation data. This large amount of Welsh translation data is surprising, ranking 4th overall for translation data and ahead of some of the most spoken languages in the world like French, Spanish, and Russian. Inspection shows the majority of supposedly Welsh translation data is actually English audio with English captions where the English audio was mis-classified as Welsh by the language identification system, resulting in it being included as translation training data rather transcription data according to our dataset creation rules.

为了对更广泛的语言集进行额外分析, 我们还将Fleurs(语音识别数据集)重新用作翻译数据集。由于每种语言都有相同的句子, 所以我们使用英语转录本作为参考译文。在图4中, 我们可视化了每种语言的翻译训练数据量与Fleurs上的零样本BLEU分数之间的相关性。虽然随着训练数据的增加有明显的改善趋势, 但平方相关系数远低于语音识别的0.83, 仅为0.24。我们怀疑这部分是由于音频语言识别中的错误导致的噪声较大的训练数据造成的。例如, 威尔士语(CY)是一个异常值, 尽管据推测有9000小时的翻译数据, 但仅在13个BLEU时的表现远低于预期。大量的威尔士语翻译数据令人惊讶, 在翻译数据上排名第四, 领先于法语、西班牙语和俄语等世界上最常用的语言。检查显示, 大多数所谓的威尔士语翻译数据实际上是带有英语字幕的英语音频, 其中英语音频被语言识别系统误分类为威尔士语, 导致根据我们的数据集创建规则, 它被包括为翻译训练数据而不是转录数据。

Figure 5. WER on LibriSpeech test-clean as a function of SNR under additive white noise (left) and pub noise (right). The accuracy of LibriSpeech-trained models degrade faster than the best Whisper model (⋆). NVIDIA STT models (*) perform best under low noise but are outperformed by Whisper under high noise (SNR \< 10 dB). The second-best model under low noise (▼) is fine-tuned on LibriSpeech only and degrades even more quickly. 
图5。LibriSpeech测试中的WER在加性白噪声(左)和公共噪声(右)下作为SNR的函数。LibriSpeech训练模型的精度比最佳Whisper模型(⋆)下降得更快。NVIDIA STT模型(*)在低噪声下表现最好, 但在高噪声(SNR<10 dB)下优于Whisper。低噪声下的第二最佳模型(▼) 仅在LibriSpeech上微调, 降级速度更快。

### 3.6. Language Identification
To evaluate language identification, we use the Fleurs dataset (Conneau et al., 2022). The zero-shot performance of Whisper is not competitive with prior supervised work here and underperforms the supervised SOTA by 13.6%. However, Whisper is heavily disadvantaged for language identification on Fleurs, since the Whisper dataset contains no training data for 20 of the 102 languages in Fleurs, upperbounding accuracy at 80.4%. On the 82 overlapping languages the best Whisper model achieves 80.3% accuracy.

为了评估语言识别, 我们使用Fleurs数据集(Conneau et al., 2022)。Whisper的零样本性能与之前的监督工作相比没有竞争力, 比监督的SOTA低13.6%。然而, Whisper在Fleurs上的语言识别方面处于严重劣势, 因为Whisper数据集不包含Fleurs中102种语言中的20种的训练数据, 在82种重叠语言中, 最佳Whisper模型的精度达到80.3%。

### 3.7. Robustness to Additive Noise  加性噪声
We tested the noise robustness of Whisper models and 14 LibriSpeech-trained models by measuring the WER when either white noise or pub noise from the Audio Degradation Toolbox (Mauch & Ewert, 2013) was added to the audio. The pub noise represents a more natural noisy environment with ambient noise and indistinct chatter typical in a crowded restaurant or a pub. Among the 14 models, twelve are pre-trained and/or fine-tuned on LibriSpeech, and the other two are NVIDIA STT models trained on a mixture dataset similar to prior work like SpeechStew that includes LibriSpeech. The level of additive noise corresponding to a given signal-to-noise ratio (SNR) is calculated based on the signal power of individual examples. Figure 5 shows how the ASR performance degrades as the additive noise becomes more intensive. There are many models that outperform our zero-shot performance under low noise (40 dB SNR), which is unsurprising given those models are trained primarily on LibriSpeech, but all models quickly degrade as the noise becomes more intensive, performing worse than the Whisper model under additive pub noise of SNR below 10 dB. This showcases Whisper’s robustness to noise, especially under more natural distribution shifts like the pub noise.

我们通过测量音频降级工具箱(Mauch&Ewert, 2013)中的白噪声或酒吧噪声添加到音频中时的WER, 测试了Whisper模型和14个LibriSpeech训练模型的噪声稳健性。酒吧的噪音代表了一种更自然的嘈杂环境, 在拥挤的餐厅或酒吧中, 环境噪音和模糊的交谈是典型的。在这14个模型中, 有12个是在LibriSpeech上预训练和/或微调的, 另外两个是在混合数据集上训练的NVIDIA STT模型, 类似于之前的工作, 如SpeechStew, 其中包括LibriSpeech。基于各个样本的信号功率计算与给定信噪比(SNR)相对应的附加噪声水平。图5显示了ASR性能如何随着附加噪声的增加而降低。有许多模型在低噪声(40 dB SNR)下表现优于我们的零炮性能, 这并不奇怪, 因为这些模型主要在LibriSpeech上训练, 但随着噪声变得更加强烈, 所有模型都会迅速退化, 在SNR低于10 dB的加性发布噪声下表现比Whisper模型更差。这展示了Whisper对噪音的稳健性, 尤其是在酒吧噪音等更自然的分布变化下。

### 3.8. Long-form Transcription 长格式转录
Whisper models are trained on 30-second audio chunks and cannot consume longer audio inputs at once. This is not a problem with most academic datasets comprised of short utterances but presents challenges in real-world applications which often require transcribing minutes- or hours-long audio. We developed a strategy to perform buffered transcription of long audio by consecutively transcribing 30-second segments of audio and shifting the window according to the timestamps predicted by the model. We observed that it is crucial to have beam search and temperature scheduling based on the repetitiveness and the log probability of the model predictions in order to reliably transcribe long audio. The full procedure is described in Section 4.5.

Whisper模型在30秒音频块上进行训练, 不能同时消耗更长的音频输入。这对于大多数由简短话语组成的学术数据集来说并不是问题, 但在现实世界的应用中却存在挑战, 因为这些应用通常需要转录几分钟或几小时长的音频。我们开发了一种策略, 通过连续转录30秒的音频片段并根据模型预测的时间戳移动窗口来执行长音频的缓冲转录。我们观察到, 为了可靠地转录长音频, 基于模型预测的重复性和对数概率进行波束搜索和温度调度至关重要。第4.5节描述了整个程序。

We evaluate the long-form transcription performance on seven datasets consisting of speech recordings of various lengths and recording conditions, to cover as diverse a data distribution as possible. These include a long-form adaptation of TED-LIUM3 (Hernandez et al., 2018) concatenated so that each example is a full-length TED talk, a collection of jargon-laden segments taken from The Late Show with Stephen Colbert (Meanwhile), sets of videos/podcasts that has been used as ASR benchmarks in online blogs (Rev16 and Kincaid46), recordings of earnings calls (Del Rio et al., 2021), and the full-length interviews from the Corpus of Regional African American Language (CORAAL) (Gunter et al., 2021). Full details about the long-form datasets can be found in Appendix A.

我们评估了由不同长度和记录条件的语音记录组成的七个数据集上的长格式转录性能, 以尽可能涵盖不同的数据分布。其中包括TED-LIUM3的长篇改编(Hernandez et al., 2018), 每个例子都是一个完整的TED演讲, 一组取自Stephen Colbert的《晚秀》(The Late Show)的术语片段(同时), 在线博客中用作ASR基准的视频/播客集(Rev16和Kincaid46), 盈利电话录音(Del Rio et al., 2021), 以及来自地区非裔美国人语言语料库(CORAAL)的完整访谈(Gunter et al., 2021)。有关长格式数据集的详情, 请参见附录A。

We compare the performance with open-source models as well as 4 commercial ASR services. The results are summarized in Figure 6, showing the distribution of word error rates from Whisper and the 4 commercial ASR services, as well as the NVIDIA STT Conformer-CTC Large model from the NeMo toolkit (Kuchaiev et al., 2019) which performed the best among the open-source models. All commercial ASR services are queried using their default English transcription settings as of September 1st, 2022, and for the NVIDIA STT model we used their buffered inference implementation in the FrameBatchASR class to enable long-form transcription. The results show that Whisper performs better than the compared models on most datasets, especially on the Meanwhile dataset which is heavy with uncommon words. Additionally, we note the possibility that some of the commercial ASR systems have been trained on some of these publicly available datasets, and therefore these results may not be accurately reflecting the relative robustness of the systems.

我们将其性能与开源模型以及4个商用ASR服务进行了比较。结果汇总在图6中, 显示了Whisper和4个商业ASR服务的单词错误率分布, 以及NeMo工具包中的NVIDIA STT Conformer CTC Large模型(Kuchaiev et al., 2019), 该模型在开源模型中表现最好。截至2022年9月1日, 所有商业ASR服务都使用其默认英语转录设置进行查询, 对于NVIDIA STT模型, 我们在FrameBatchASR类中使用了其缓冲推理实现, 以启用长格式转录。结果表明, Whisper在大多数数据集上的表现都优于比较模型, 尤其是在同时数据集上, 该数据集包含大量不常见的单词。此外, 我们注意到一些商业ASR系统可能已经在这些公开可用的数据集上进行了训练, 因此这些结果可能无法准确反映系统的相对稳健性。

Figure 6. Whisper is competitive with state-of-the-art commercial and open-source ASR systems in long-form transcription. The distribution of word error rates from six ASR systems on seven long-form datasets are compared, where the input lengths range from a few minutes to a few hours. The boxes show the quartiles of per-example WERs, and the per-dataset aggregate WERs are annotated on each box. Our model outperforms the best open source model (NVIDIA STT) on all datasets, and in most cases, commercial ASR systems as well.
图6。Whisper与最先进的商业和开源ASR系统在长格式转录方面具有竞争力。比较了七个长格式数据集上六个ASR系统的单词错误率分布, 其中输入长度从几分钟到几小时不等。这些框显示了每个样本WER的四分位数, 每个框上都标注了每个数据集的聚集WER。我们的模型在所有数据集上都优于最好的开源模型(NVIDIA STT), 在大多数情况下, 商业ASR系统也是如此。

### 3.9. Comparison with Human Performance
Because of ambiguous or indistinct speech as well as labeling errors, there are different levels of irreducible error in each dataset, and with WER metrics from ASR systems alone it is difficult to make sense of how much room for improvement exists in each dataset. To quantify how close Whisper’s performance is to the human performance, we selected 25 recordings from the Kincaid46 dataset and used 5 services to obtain transcripts produced by professional transcribers, among which one provides computer-assisted transcription and the other four are entirely human-transcribed. The audio selection covers various recording conditions such as scripted and unscripted broadcast, telephone and VoIP calls, and meetings. Figure 7 shows the distribution of per-example WERs and aggregate WER across the 25 recordings, where the computer-assisted service has the lowest aggregate WER that is 1.15% point better than Whisper’s, and the pure-human performance is only a fraction of a percentage point better than Whisper’s. These results indicate that Whisper’s English ASR performance is not perfect but very close to human-level accuracy.

由于模糊或不清晰的语音以及标注错误, 每个数据集中存在不同程度的不可约错误, 并且仅使用ASR系统的WER度量, 很难理解每个数据集中有多少改进空间。为了量化Whisper的表现与人类的表现有多接近, 我们从Kincaid46数据集中选择了25段录音, 并使用5种服务获取专业转录师制作的转录本, 其中一种提供计算机辅助转录, 其他四种完全由人类转录。音频选择涵盖各种录制条件, 如脚本和非脚本广播、电话和VoIP通话以及会议。图7显示了25段录音中每个样本的WER和总WER的分布情况, 其中计算机辅助服务的总WER最低, 比Whisper的好1.15%, 纯人类的表现只比Whispr的好一个百分点。这些结果表明, Whisper的英语ASR性能并不完美, 但非常接近人类水平的准确性。

## 4. Analysis and Ablations
### 4.1. Model Scaling
A large amount of the promise in weakly supervised training approaches is their potential to use datasets much larger than those in traditional supervised learning. However, this comes with the cost of using data that is possibly much noisier and lower quality than gold-standard supervision. A concern with this approach is that although it may look promising to begin with, the performance of models trained on this kind of data may saturate at the inherent quality level of the dataset, which could be far below human level. A related concern is that as capacity and compute spent training on the dataset increases, models may learn to exploit the idiosyncrasies of the dataset, and their ability to generalize robustly to out-of-distribution data could even degrade.

弱监督训练方法的一大优势是其使用数据集的潜力远大于传统监督学习中的数据集。然而, 这带来了使用数据的成本, 这些数据可能比金标准监管更嘈杂, 质量更低。这种方法的一个问题是, 尽管一开始看起来很有希望, 但基于这类数据训练的模型的性能可能在数据集的固有质量水平上饱和, 这可能远远低于人类水平。一个相关的担忧是, 随着数据集上的容量和计算训练的增加, 模型可能会学习利用数据集的特性, 它们对分布外数据的稳健泛化能力甚至会降低。

Figure 7. Whisper’s performance is close to that of professional human transcribers. This plot shows the WER distributions of 25 recordings from the Kincaid46 dataset transcribed by Whisper, the same 4 commercial ASR systems from Figure 6 (A-D), one computer-assisted human transcription service (E) and 4 human transcription services (F-I). The box plot is superimposed with dots indicating the WERs on individual recordings, and the aggregate WER over the 25 recordings are annotated on each box. 
图7。Whisper的表现接近于专业的人类转录员。该图显示了Whisper转录的Kincaid46数据集的25条录音的WER分布, 图6(A-D)中的4个商业ASR系统, 一个计算机辅助人类转录服务(E)和4个人类转录服务。方框图上叠加有点, 表示单个记录的WER, 25个记录的总WER在每个方框上标注。

To check whether this is the case, we study the zero-shot generalization of Whisper models as a function of the model size. Our analysis is summarized in Figure 8. With the exception of English speech recognition, performance continues to increase with model size across multilingual speech recognition, speech translation, and language identification. The diminishing returns for English speech recognition could be due to saturation effects from approaching humanlevel performance as analysis in Section 3.9 suggests.

为了检验这种情况是否属实, 我们研究了Whisper模型的零炮泛化作为模型大小的函数。我们的分析总结在图8中。除了英语语音识别之外, 在多语言语音识别、语音翻译和语言识别中, 性能随着模型大小的增加而不断提高。正如第3.9节中的分析所表明的, 英语语音识别的回报递减可能是由于接近人类水平的表现所产生的饱和效应。

### 4.2. Dataset Scaling
At 680,000 hours of labeled audio, the Whisper dataset is one of the largest ever created in supervised speech recognition. Exactly how important is the raw dataset size to Whisper’s performance? To study this, we trained a series of medium-sized models on subsampled versions of the dataset which are 0.5%, 1%, 2%, 4%, and 8% of the full dataset size and compared their performance with the same medium-sized model trained on the whole dataset. Early stopping based on the validation loss was used to select model checkpoints for each dataset size. Evaluation was performed on an exponential moving average estimate of the parameters (Polyak & Juditsky, 1992) using a smoothing rate of 0.9999 to help reduce the effect of the learning rate not fully decaying to zero for the models trained on the subsampled datasets due to early stopping. Performance on English and multilingual speech recognition and X→en translation is reported in Table 6.

Whisper数据集拥有68万小时的标签音频, 是有史以来最大的有监督语音识别数据集之一。原始数据集大小对Whisper的性能到底有多重要？为了研究这一点, 我们在全数据集大小的0.5%、1%、2%、4%和8%的数据集的二次采样版本上训练了一系列中型模型, 并将它们的性能与在整个数据集上训练的相同中型模型进行了比较。基于验证丢失的提前停止用于为每个数据集大小选择模型检查点。使用0.9999的平滑率对参数的指数移动平均估计值进行评估(Polyak&Juditsky, 1992), 以帮助减少由于早期停止而在二次采样数据集上训练的模型的学习率未完全衰减到零的影响。英语和多语言语音识别性能和X→表6中报告了英语翻译。

Figure 8. Zero-shot Whisper performance scales reliably across tasks and languages with increasing model size. Lightly shaded lines represent individual datasets or languages, showing that performance is more varied than the smooth trends in aggregate performance. Large V2 distinguished with a dashed orange line since it includes several changes that are not present for the smaller models in this analysis.
图8。Zero-shot Whisper性能随着模型大小的增加而在任务和语言之间可靠地扩展。浅色阴影线表示单个数据集或语言, 表明性能比聚合性能的平滑趋势更为多样。大V2以橙色虚线区分, 因为它包括本分析中较小模型不存在的几个变化。

Table 6. Performance improves with increasing dataset size. English speech recognition performance refers to an average over 12 datasets while the Multilingual speech recognition reports performance on the overlapping subset of languages in Fleurs and X→en translation reports average BLEU on CoVoST2. Dataset size reported in hours.
表6。性能随着数据集大小的增加而提高。英语语音识别性能指的是平均超过12个数据集, 而多语言语音识别报告在Fleurs和X语言的重叠子集上的性能→en translation报告CoVoST2上的平均BLEU。以小时为单位报告的数据集大小。

All increases in the dataset size result in improved performance on all tasks, although we see significant variability in improvement rates across tasks and sizes. Performance improves rapidly on English speech recognition from 3,000 to 13,000 hours and then slows down noticeably between 13,000 and 54,000 hours. Using the full dataset, which corresponds to another 12.5× increase in size results in only a further 1 point drop in WER. This mirrors the diminishing returns observed with model size scaling for English speech recognition and could similarly be explained by saturation effects when approaching human-level performance.

数据集大小的所有增加都会提高所有任务的性能, 尽管我们看到不同任务和大小的改进率存在显著差异。英语语音识别的性能从3000到13000小时快速提高, 然后在13000到54000小时之间显著降低。使用完整的数据集(对应于大小的另一个12.5倍增长), WER仅进一步下降1个点。这反映了英语语音识别的模型大小缩放所观察到的回报递减, 并且可以通过接近人类水平的表现时的饱和效应来解释。

Improvements in WER follow a power-law trend for multilingual speech recognition till 54,000 hours and then deviate from this trend, improving only a further 7 points when increasing to the full dataset size. For X→en translation, performance is practically zero when training on 7,000 hours of audio or less, and then follows a roughly log-linear improvement trend till 54,000 hours before also showing diminishing returns when further scaling to the full dataset size.

WER的改进遵循多语言语音识别的幂律趋势, 直到54000小时, 然后偏离这一趋势, 当增加到完整数据集大小时, 仅进一步提高7个点。对于X→换言之, 当在7000小时或更短的音频上进行训练时, 性能几乎为零, 然后遵循大致对数线性的改善趋势, 直到54000小时, 当进一步扩展到完整的数据集大小时, 也显示出递减的回报。

The general trend across tasks of diminishing returns when moving from 54,000 hours to our full dataset size of 680,000 hours could suggest that the current best Whisper models are under-trained relative to dataset size and performance could be further improved by a combination of longer training and larger models. It could also suggest that we are nearing the end of performance improvements from dataset size scaling for speech recognition. Further analysis is needed to characterize “scaling laws” for speech recognition in order to decided between these explanations.

当从54000小时移动到680000小时的完整数据集大小时, 整个任务的总体趋势是收益递减, 这可能表明当前最好的Whisper模型相对于数据集大小而言训练不足, 并且可以通过更长的训练和更大的模型的组合来进一步提高性能。这也可能表明, 从数据集大小缩放到语音识别, 我们的性能改进已经接近尾声。需要进一步的分析来表征语音识别的“缩放定律”, 以便在这些解释之间做出决定。

### 4.3. Multitask and Multilingual Transfer
A potential concern with jointly training a single model on many tasks and languages is the possibility of negative transfer where interference between the learning of several tasks results in performance worse than would be achieved by training on only a single task or language. To investigate whether this is occurring, we compared the performance of models trained on just English speech recognition with our standard multitask and multilingual training setup and measured their average performance across our suite of zeroshot English speech recognition benchmarks. We adjust for the amount of FLOPs spent training on the task of English speech recognition as only 65% of compute is spent on this task in a joint training setup; analysis would otherwise be confounded by under-training on the task when compared to a same-sized English-only model.

在多个任务和语言上联合训练单个模型的一个潜在问题是负迁移的可能性, 在这种情况下, 多个任务的学习之间的干扰会导致性能比仅在单个任务或语言上训练更差。为了调查这种情况是否发生, 我们将仅针对英语语音识别训练的模型的性能与我们的标准多任务和多语言训练设置进行了比较, 并测量了它们在我们的一套零距离英语语音识别基准中的平均性能。我们调整了用于英语语音识别任务的FLOP训练量, 因为在联合训练设置中, 只有65%的计算用于该任务; 否则, 与同一规模的纯英语模型相比, 分析会因任务训练不足而变得混乱。

Our results visualized in Figure 9 show that for small models trained with moderate amounts of compute, there is indeed negative transfer between tasks and languages: joint models underperform English-only models trained for the same amount of compute. However, multitask and multilingual models scale better and for our largest experiments outperform their English-only counterparts demonstrating positive transfer from other tasks. For our largest experiments, joint models also slightly outperform English-only models even when not adjusting for compute spent per task.

我们在图9中看到的结果表明, 对于使用中等计算量训练的小模型, 任务和语言之间确实存在负迁移：联合模型的表现低于使用相同计算量训练过的纯英语模型。然而, 多任务和多语言模型的规模更大, 对于我们最大的实验来说, 其表现优于仅英语的模型, 证明了其他任务的正向迁移。在我们最大的实验中, 联合模型也稍微优于纯英语模型, 即使不考虑每项任务的计算开销。

Figure 9. Multitask and multilingual transfer improves with scale. For small models, performance on English speech recognition degrades when trained jointly in a multitask and multilingual setup. However, multilingual and multitask models benefit more from scale and eventually outperform models trained on English data only. 95% bootstrap estimate confidence intervals are shown. 

图9。多任务和多语言迁移随着规模的增加而提高。对于小型模型, 当在多任务和多语言设置中联合训练时, 英语语音识别的性能会下降。然而, 多语言和多任务模型从规模中受益更多, 并最终超过仅使用英语数据训练的模型。显示了95%的自举估计置信区间。

### 4.4. Text Normalization
Since we developed our text normalization jointly with Whisper to discount innocuous word errors, there is a risk that our normalizer is overfitted to fixing Whisper’s peculiarities rather than addressing general variation in transcription. To check this, we compared the performance of Whisper using our normalizer versus an independently developed one from the FairSpeech project (Koenecke et al., 2020). In Figure 10, we visualize the differences. On most datasets the two normalizers perform similarly, without significant differences in WER reduction between Whisper and compared open-source models, while on some datasets, namely WSJ, CallHome, and Switchboard, our normalizer reduces the WER of Whisper models’ significantly more. The differences in reduction can be traced down to different formats used by the ground truth and how the two normalizers are penalizing them. For example, in CallHome and Switchboard, our standardizer did not penalize differences in common English contractions such as “you’re” versus “you are”, and in WSJ, our normalizer standardized the written and spoken forms of numerical and monetary expressions, such as “sixty-eight million dollars” versus “$68 million”. 

由于我们与Whisper联合开发了文本规范化, 以消除无害的单词错误, 因此存在一种风险, 即我们的规范化器过度适合于修复Whisper的特性, 而不是解决转录中的一般差异。为了验证这一点, 我们将使用我们的归一化器的Whisper性能与FairSpeech项目中独立开发的归一化器进行了比较(Koenecke et al., 2020)。在图10中, 我们将这些差异可视化。在大多数数据集上, 两个标准化器的性能相似, Whisper和比较的开源模型之间的WER降低没有显著差异, 而在一些数据集上(即WSJ、CallHome和Switchboard), 我们的标准化器显著降低了Whisper模型的WER。减少的差异可以追溯到地面真相使用的不同格式, 以及两个归一化器如何惩罚它们。例如, 在CallHome和Switchboard中, 我们的标准化器没有惩罚常见英语缩略词的差异, 例如“you are”和“you are”, 而在WSJ中, 我们标准化器标准化了数字和货币表达的书面和口头形式, 例如“6800万美元”和“6800美元”。

Figure 10. On most datasets, our text normalizer has similar effect on reducing WERs between Whisper models and other open-source models, compared to FairSpeech’s normalizer. For each dataset, the boxplot shows the distribution of relative WER reduction across different models in our eval suite, showing that using our text normalizer generally results in lower WERs than FairSpeech’s. On a few datasets our normalizer reduces WER significantly and more so for Whisper models, such as CallHome and Switchboard which have many contractions in the ground truth and WSJ which contains many numerical expressions. 
图10。在大多数数据集上, 与FairSpeech的规范化器相比, 我们的文本规范化器在减少Whisper模型和其他开源模型之间的WER方面具有类似的效果。对于每个数据集, 方框图显示了eval套件中不同模型之间相对WER减少的分布, 表明使用文本归一化器通常会导致比FairSpeech更低的WER。在一些数据集上, 我们的归一化器显著降低了WER, 而对于Whisper模型, 如CallHome和Switchboard, 它们在地面真相中有许多收缩, WSJ包含许多数值表达式。

### 4.5. Strategies for Reliable Long-form Transcription
Transcribing long-form audio using Whisper relies on accurate prediction of the timestamp tokens to determine the amount to shift the model’s 30-second audio context window by, and inaccurate transcription in one window may negatively impact transcription in the subsequent windows. We have developed a set of heuristics that help avoid failure cases of long-form transcription, which is applied in the results reported in sections 3.8 and 3.9. First, we use beam search with 5 beams using the log probability as the score function, to reduce repetition looping which happens more frequently in greedy decoding. We start with temperature 0, i.e. always selecting the tokens with the highest probability, and increase the temperature by 0.2 up to 1.0 when either the average log probability over the generated tokens is lower than −1 or the generated text has a gzip compression rate higher than 2.4. Providing the transcribed text from the preceding window as previous-text conditioning when the applied temperature is below 0.5 further improves the performance. We found that the probability of the \<|nospeech|> token alone is not sufficient to distinguish a segment with no speech, but combining the no-speech probability threshold of 0.6 and the average log-probability threshold of −1 makes the voice activity detection of Whisper more reliable. Finally, to avoid a failure mode where the model ignores the first few words in the input, we constrained the initial timestamp token to be between 0.0 and 1.0 second. Table 7 shows that adding each of the interventions above incrementally reduces the WER overall, but not evenly across the dataset. These heuristics serve as a workaround for the noisy predictions of the model, and more research would be needed to further improve the reliability of long-form decoding.

使用Whisper转录长格式音频依赖于时间戳令牌的准确预测, 以确定将模型的30秒音频上下文窗口移动的量, 一个窗口中的不准确转录可能会对后续窗口中的转录产生负面影响。我们开发了一套启发式方法, 帮助避免长形式转录的失败案例, 这在第3.8和3.9节报告的结果中得到了应用。首先, 我们使用对数概率作为得分函数对5个波束进行波束搜索, 以减少贪婪解码中更频繁发生的重复循环。我们从温度0开始, 即始终选择概率最高的令牌, 并在生成的令牌的平均对数概率低于−1或生成的文本的gzip压缩率高于2.4时, 将温度提高0.2至1.0。当施加的温度低于0.5时, 提供来自先前窗口的转录文本作为先前文本调节进一步提高了性能。我们发现, 单独使用\<|nospeech|>令牌的概率不足以区分没有语音的片段, 但结合0.6的无语音概率阈值和−1的平均对数概率阈值, Whisper的语音活动检测更加可靠。最后, 为了避免模型忽略输入中的前几个单词的失败模式, 我们将初始时间戳令牌限制在0.0到1.0秒之间。表7显示, 添加上述每一项干预措施会逐步降低总体WER, 但不会在整个数据集中均匀降低。这些启发式方法作为模型的噪声预测的解决方法, 需要进行更多的研究以进一步提高长格式解码的可靠性。

Table 7. Long-form transcription performance improves incrementally as additional decoding heuristics are employed. Details on each intervention are described in Section 4.5. 
表7。随着使用额外的解码启发式, 长格式转录性能逐渐提高。各干预措施的详情见第4.5节。

## 5. Related Work
Scaling Speech Recognition A consistent theme across speech recognition research has been documenting the benefits of scaling compute, models, and datasets. Early work applying deep learning to speech recognition found improved performance with model depth and size and leveraged GPU acceleration to make training these larger models tractable (Mohamed et al., 2009). Further research demonstrated that the benefit of deep learning approaches to speech recognition increased with dataset size, improving from being only competitive with prior GMM-HMM systems when using just 3 hours of TIMIT training data for phone recognition to achieving a 30% word error rate reduction when trained on the 2,000 hour Switchboard dataset (Seide et al., 2011). Liao et al. (2013) is an early example of leveraging weakly supervised learning to increase the size of a deep learning based speech recognition dataset by over 1,000 hours. These trends continued with Deep Speech 2 (Amodei et al., 2015) being a notable system developing high-throughput distributed training across 16 GPUs and scaling to 12,000 hours of training data while demonstrating continuing improvements at that scale. By leveraging semi-supervised pre-training, Narayanan et al. (2018) were able to grow dataset size much further and study training on 162,000 hours of labeled audio. More recent work has explored billion-parameter models (Zhang et al., 2020) and using up to 1,000,000 hours of training data (Zhang et al., 2021).

缩放语音识别语音识别研究的一个一致主题是记录缩放计算、模型和数据集的好处。早期将深度学习应用于语音识别的工作发现, 模型深度和大小提高了性能, 并利用GPU加速使训练这些更大的模型变得易于处理(Mohamed et al., 2009)。进一步的研究表明, 深度学习方法对语音识别的益处随着数据集的大小而增加, 从仅使用3小时的TIMIT训练数据进行电话识别时仅与现有的GMM-HMM系统竞争, 到在2000小时的开关板数据集上训练时实现30%的单词错误率降低(Seide et al., 2011)。Liao et al., (2013)是利用弱监督学习将基于深度学习的语音识别数据集的大小增加1000多小时的早期样本。随着深度语音2(Amodei et al., 2015)成为一个值得注意的系统, 该系统在16个GPU上开发高吞吐量分布式训练, 并扩展到12000小时的训练数据, 同时证明了在该规模上的持续改进, 这些趋势仍在继续。通过利用半监督预训练, Narayanan et al., (2018)能够进一步增加数据集的大小, 并在162000小时的标注音频上研究训练。最近的工作探索了十亿参数模型(Zhang et al., 2020), 并使用了多达1000000小时的训练数据(Zhang et al., 2021)。

Multitask Learning Multitask learning (Caruana, 1997) has been studied for a long time. In speech recognition, multi-lingual models have been explored for well over a decade (Schultz & Kirchhoff, 2006). An inspirational and foundational work in NLP exploring multi-task learning with a single model is Collobert et al. (2011). Multitask learning in the sequence-to-sequence framework (Sutskever et al., 2014) using multiple encoders and decoders was investigated in Luong et al. (2015). The use of language codes with a shared encoder/decoder architecture was first demonstrated for machine translation by Johnson et al. (2017), removing the need for separate encoders and decoders. This approach was simplified further into the “text-to-text” framework of McCann et al. (2018) and popularized by its success with large transformer language models in the work of Radford et al. (2019) and Raffel et al. (2020). Toshniwal et al. (2018) demonstrated jointly training a modern deep learning speech recognition system on several languages with a single model, and Pratap et al. (2020a) scaled this line of work significantly to 50 languages with a billion-parameter model. MUTE (Wang et al., 2020c) and mSLAM (Bapna et al., 2022) studied joint training over both text and speech language tasks, demonstrating transfer between them.

多任务学习多任务学习(Caruana, 1997)已经被研究了很长时间。在语音识别中, 多语言模型已经被探索了十多年(Schultz&Kirchhoff, 2006)。Collobert et al., (2011)在NLP中进行了一项鼓舞人心的基础性工作, 即利用单一模型探索多任务学习。Luong et al., (2015)研究了使用多个编码器和解码器的序列到序列框架中的多任务学习(Sutskever et al., 2014)。Johnson et al., (2017)首次在机器翻译中演示了具有共享编码器/解码器架构的语言代码的使用, 从而消除了对单独编码器和解码器的需求。这种方法被进一步简化为McCann et al., 的“文本到文本”框架。Toshniwal et al., (2018)展示了用一个模型在几种语言上联合训练现代深度学习语音识别系统, Pratap et al., (2020a)用十亿个参数模型将这一工作线大幅扩展到50种语言。MUTE(Wang et al., 2020c)和mSLAM(Bapna et al., 2022)研究了文本和语音语言任务的联合训练, 证明了它们之间的迁移。

Robustness The question of how effectively models transfer and how robust they are to distribution shift and other types of perturbations has long been studied and is actively being researched across many fields of machine learning. Torralba & Efros (2011) highlighted the lack of generalization of machine learning models between datasets over a decade ago. Many other works have shown and continually reiterated how despite high performance on IID test sets, machine learning models can still make many mistakes when evaluated in even slightly different settings (Lake et al., 2017; Jia & Liang, 2017; Alcorn et al., 2019; Barbu et al., 2019; Recht et al., 2019). More recently, Taori et al. (2020) studied the robustness of image classification models, and Miller et al. (2020) investigated this for question-answering models. A key finding has been that multi-domain training increases robustness and generalization as discussed in the Introduction. This finding has been replicated across many fields in addition to speech recognition including NLP (Hendrycks et al., 2020) and computer vision (Radford et al., 2021).

稳健性模型传递的有效性以及它们对分布迁移和其他类型扰动的稳健性问题早已被研究, 并正在机器学习的许多领域中积极研究。Torralba和Efros(2011)强调了十年前数据集之间缺乏机器学习模型的通用性。许多其他研究表明并不断重申, 尽管在IID测试集上表现优异, 但机器学习模型在甚至稍微不同的设置下进行评估时仍然会犯许多错误(Lake et al., 2017; Jia&Liang, 2017; Alcorn et al., 2019; Barbu et al., 2019;Recht et al., 2019)。最近, Taori et al., (2020)研究了图像分类模型的稳健性, Miller et al., (2020)针对问答模型对此进行了研究。一个关键发现是, 如引言中所述, 多域训练提高了稳健性和通用性。除了语音识别(NLP)(Hendrycks et al., 2020)和计算机视觉(Radford et al., 2021)之外, 这一发现在许多领域都得到了复制。

## 6. Limitations and Future Work
From our experimental results, analyses, and ablations, we have noted several limitations and areas for future work.

从我们的实验结果、分析和消融中, 我们注意到了几个局限性和未来的工作领域。

### Improved decoding strategies. 
As we have scaled Whisper, we have observed that larger models have made steady and reliable progress on reducing perception-related errors such as confusing similar-sounding words. Many remaining errors, particularly in long-form transcription seem more stubborn in nature and decidedly non-human/perceptual. They are a combination of failure modes of seq2seq models, language models, and text-audio alignment and include problems such as getting stuck in repeat loops, not transcribing the first or last few words of an audio segment, or complete hallucination where the model will output a transcript entirely unrelated to the actual audio. Although the decoding details discussed in Section 4.5 help significantly, we suspect fine-tuning Whisper models on a high-quality supervised dataset and/or using reinforcement learning to more directly optimize for decoding performance could help further reduce these errors.

改进的解码策略。随着我们对Whisper进行了缩放, 我们观察到更大的模型在减少与感知相关的错误(如混淆发音相似的单词)方面取得了稳定可靠的进展。许多剩余的错误, 特别是在长形式的转录中, 似乎在本质上更加顽固, 而且显然是非人类/感性的。它们是seq2seq模型、语言模型和文本音频对齐的失败模式的组合, 包括陷入重复循环、无法转录音频片段的前几个或最后几个单词或完全幻觉等问题, 其中模型将输出与实际音频完全无关的转录。尽管第4.5节中讨论的解码细节有很大帮助, 但我们怀疑在高质量监督数据集上微调Whisper模型和/或使用强化学习来更直接地优化解码性能可能有助于进一步减少这些错误。

### Increase Training Data For Lower-Resource Languages. 
As Figure 3 shows, Whisper’s speech recognition performance is still quite poor on many languages. The same analysis suggests a clear route for improvement since performance on a language is very well predicted by the amount of training data for the language. Since our pre-training dataset is currently very English-heavy due to biases of our data collection pipeline, which sourced primarily from English-centric parts of the internet, most languages have less than 1000 hours of training data. A targeted effort at increasing the amount of data for these rarer languages could result in a large improvement to average speech recognition performance even with only a small increase in our overall training dataset size.

增加低资源语言的训练数据。如图3所示, Whisper在许多语言上的语音识别性能仍然很差。同样的分析表明了一个明显的改进途径, 因为语言的表现可以通过语言的训练数据量来很好地预测。由于我们的数据收集管道(主要来自互联网中以英语为中心的部分)存在偏见, 我们的训练前数据集目前非常重英语, 因此大多数语言的训练数据不足1000小时。针对这些稀有语言增加数据量的目标努力可能会大大提高平均语音识别性能, 即使我们的总体训练数据集大小只增加了一小部分。

Studying fine-tuning In this work, we have focused on the robustness properties of speech processing systems and as a result only studied the zero-shot transfer performance of Whisper. While this is a crucial setting to study due to it being representative of general reliability, for many domains where high-quality supervised speech data does exist, it is likely that results can be improved further by fine-tuning. An additional benefit of studying fine-tuning is that it allows for direct comparisons with prior work since it is a much more common evaluation setting.

研究微调在这项工作中, 我们专注于语音处理系统的稳健性属性, 因此只研究了Whisper的零样本迁移性能。虽然这是一个关键的研究设置, 因为它代表了一般的可靠性, 但对于存在高质量监督语音数据的许多领域, 可能通过微调可以进一步改善结果。研究微调的另一个好处是, 它允许与以前的工作进行直接比较, 因为这是一种更常见的评估设置。

### Studying the impact of Language Models on Robustness. 
As argued in the introduction, we suspect that Whisper’s robustness is partially due to its strong decoder, which is an audio conditional language model. It’s currently unclear to what degree the benefits of Whisper stem from training its encoder, decoder, or both. This could be studied by either ablating various design components of Whisper, such as training a decoder-less CTC model, or by studying how the performance of existing speech recognition encoders such as wav2vec 2.0 change when used together with a language model.

研究语言模型对稳健性的影响。正如引言中所述, 我们怀疑Whisper的稳健性部分是由于其强大的解码器, 这是一个音频条件语言模型。目前还不清楚Whisper在多大程度上得益于训练编码器、解码器或两者。这可以通过消除Whisper的各种设计组件来研究, 例如训练无解码器的CTC模型, 或者通过研究现有语音识别编码器(如wav2vec 2.0)在与语言模型一起使用时的性能如何变化。

### Adding Auxiliary Training Objectives. 
Whisper departs noticeably from most recent state-of-the-art speech recognition systems due to the lack of unsupervised pre-training or self-teaching methods. While we have not found them necessary to achieve good performance, it is possible that the results could be further improved by incorporating this.

增加辅助训练目标。由于缺乏无监督的预训练或自学方法, Whisper与最新的最先进的语音识别系统明显不同。虽然我们发现它们对于实现良好的性能是不必要的, 但通过结合这一点, 结果可能会进一步改善。

## 7. Conclusion
Whisper suggests that scaling weakly supervised pretraining has been underappreciated so far in speech recognition research. We achieve our results without the need for the self-supervision and self-training techniques that have been a mainstay of recent large-scale speech recognition work and demonstrate how simply training on a large and diverse supervised dataset and focusing on zero-shot transfer can significantly improve the robustness of a speech recognition system.

Whisper认为, 到目前为止, 在语音识别研究中, 缩放弱监督预训练尚未得到充分重视。我们实现了我们的结果, 而不需要作为最近大规模语音识别工作的支柱的自监督和自我训练技术, 并证明了简单地在大型和多样化的监督数据集上进行训练并专注于零样本迁移可以显著提高语音识别系统的稳健性。

## ACKNOWLEDGMENTS
We’d like to thank the millions of people who were involved in creating the data used by Whisper. We’d also like to thank Nick Ryder, Will Zhuk, and Andrew Carr for the conversation on the waterfall hike that inspired this project. We are also grateful to the Acceleration and Supercomputing teams at OpenAI for their critical work on software and hardware infrastructure this project used. We’d also like to thank Pamela Mishkin for advising the project from a policy perspective. Finally, we are grateful to the developers of the many software packages used throughout this project including, but not limited, to Numpy (Harris et al., 2020), SciPy (Virtanen et al., 2020), ftfy (Speer, 2019), PyTorch (Paszke et al., 2019), pandas (pandas development team, 2020), and scikit-learn (Pedregosa et al., 2011).

我们要感谢参与创建Whisper使用的数据的数百万人。我们还要感谢Nick Ryder、Will Zhuk和Andrew Carr在瀑布徒步旅行中的对话, 这激发了这个项目。我们还感谢OpenAI的加速和超级计算团队在本项目使用的软件和硬件基础设施方面所做的关键工作。我们还要感谢帕梅拉·米什金从策略角度为该项目提供建议。最后, 我们感谢整个项目中使用的许多软件包的开发人员, 包括但不限于Numpy(Harris et al., 2020)、SciPy(Virtanen et al., 2020)、ftfy(Speer, 2019)、PyTorch(Paszke et al., 2019)、panda(panda开发团队, 2020)和scikit learn(Pedregosa et al., 2011)。

## References
* Alcorn, M. A., Li, Q., Gong, Z., Wang, C., Mai, L., Ku, W.-S., and Nguyen, A. Strike (with) a pose: Neural networksare easily fooled by strange poses of familiar objects. InProceedings of the IEEE/CVF Conference on ComputerVision and Pattern Recognition, pp. 4845–4854, 2019.
* Amodei, D., Anubhai, R., Battenberg, E., Case, C., Casper,J., Catanzaro, B., Chen, J., Chrzanowski, M., Coates,A., Diamos, G., et al. Deep speech 2: end-to-end speechrecognition in english and mandarin. arxiv. arXiv preprintarXiv:1512.02595, 2015.
* Ardila, R., Branson, M., Davis, K., Henretty, M., Kohler,M., Meyer, J., Morais, R., Saunders, L., Tyers, F. M.,and Weber, G. Common voice: A massively-multilingualspeech corpus. arXiv preprint arXiv:1912.06670, 2019.
* Babu, A., Wang, C., Tjandra, A., Lakhotia, K., Xu,Q., Goyal, N., Singh, K., von Platen, P., Saraf, Y.,Pino, J., et al. XLS-R: Self-supervised cross-lingualspeech representation learning at scale. arXiv preprintarXiv:2111.09296, 2021.
* Baevski, A., Zhou, H., Mohamed, A., and Auli, M. wav2vec2.0: A framework for self-supervised learning of speechrepresentations. arXiv preprint arXiv:2006.11477, 2020.
* Baevski, A., Hsu, W.-N., Conneau, A., and Auli, M. Unsu￾pervised speech recognition. Advances in Neural Infor￾mation Processing Systems, 34:27826–27839, 2021.
* Bapna, A., Cherry, C., Zhang, Y., Jia, Y., Johnson, M.,Cheng, Y., Khanuja, S., Riesa, J., and Conneau, A. mslam:Massively multilingual joint pre-training for speech andtext. arXiv preprint arXiv:2202.01374, 2022.
* Barbu, A., Mayo, D., Alverio, J., Luo, W., Wang, C., Gut￾freund, D., Tenenbaum, J., and Katz, B. Objectnet: Alarge-scale bias-controlled dataset for pushing the lim￾its of object recognition models. Advances in neuralinformation processing systems, 32, 2019.
* Caruana, R. Multitask learning. Machine learning, 28(1):41–75, 1997.
* Chan, W., Park, D., Lee, C., Zhang, Y., Le, Q., and Norouzi,M. SpeechStew: Simply mix all available speech recogni￾tion data to train one large neural network. arXiv preprintarXiv:2104.02133, 2021.
* Chen, G., Chai, S., Wang, G., Du, J., Zhang, W.-Q.,Weng, C., Su, D., Povey, D., Trmal, J., Zhang, J.,et al. Gigaspeech: An evolving, multi-domain asr corpuswith 10,000 hours of transcribed audio. arXiv preprintarXiv:2106.06909, 2021.
* Chen, S., Wu, Y., Wang, C., Chen, Z., Chen, Z., Liu, S.,Wu, J., Qian, Y., Wei, F., Li, J., et al. Unispeech-sat: Uni￾versal speech representation learning with speaker awarepre-training. In ICASSP 2022-2022 IEEE InternationalConference on Acoustics, Speech and Signal Processing(ICASSP), pp. 6152–6156. IEEE, 2022a.
* Chen, T., Xu, B., Zhang, C., and Guestrin, C. Trainingdeep nets with sublinear memory cost. arXiv preprintarXiv:1604.06174, 2016.
* Chen, Z., Zhang, Y., Rosenberg, A., Ramabhadran, B.,Moreno, P., Bapna, A., and Zen, H. Maestro: Matchedspeech text representations through modality matching. arXiv preprint arXiv:2204.03409, 2022b.
* Child, R., Gray, S., Radford, A., and Sutskever, I. Gen￾erating long sequences with sparse transformers. arXivpreprint arXiv:1904.10509, 2019.
* Collobert, R., Weston, J., Bottou, L., Karlen, M.,Kavukcuoglu, K., and Kuksa, P. Natural language pro￾cessing (almost) from scratch. Journal of machine learn￾ing research, 12(ARTICLE):2493–2537, 2011.
* Conneau, A., Ma, M., Khanuja, S., Zhang, Y., Axelrod, V.,Dalmia, S., Riesa, J., Rivera, C., and Bapna, A. Fleurs:Few-shot learning evaluation of universal representationsof speech. arXiv preprint arXiv:2205.12446, 2022.
* Del Rio, M., Delworth, N., Westerman, R., Huang, M.,Bhandari, N., Palakapilly, J., McNamara, Q., Dong, J.,Zelasko, P., and Jett´e, M. Earnings-21: a practical bench￾mark for asr in the wild. arXiv preprint arXiv:2104.11348,2021.
* Galvez, D., Diamos, G., Torres, J. M. C., Achorn, K., Gopi,A., Kanter, D., Lam, M., Mazumder, M., and Reddi, V. J.
* The people’s speech: A large-scale diverse english speechrecognition dataset for commercial usage. arXiv preprintarXiv:2111.09344, 2021.
* Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Bren￾del, W., Bethge, M., and Wichmann, F. A. Shortcut learn￾ing in deep neural networks. Nature Machine Intelligence,2(11):665–673, 2020.
* Ghorbani, B., Firat, O., Freitag, M., Bapna, A., Krikun,M., Garcia, X., Chelba, C., and Cherry, C. Scalinglaws for neural machine translation. arXiv preprintarXiv:2109.07740, 2021.
* Griewank, A. and Walther, A. Algorithm 799: revolve: animplementation of checkpointing for the reverse or ad￾joint mode of computational differentiation. ACM Trans￾actions on Mathematical Software (TOMS), 26(1):19–45,2000.
* Gunter, K., Vaughn, C., and Kendall, T. Contextualiz￾ing/s/retraction: Sibilant variation and change in wash￾ington dc african american language. Language Variationand Change, 33(3):331–357, 2021.
* Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers,R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J.,Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., vanKerkwijk, M. H., Brett, M., Haldane, A., Fern´andez delR´ıo, J., Wiebe, M., Peterson, P., G´erard-Marchant, P.,Sheppard, K., Reddy, T., Weckesser, W., Abbasi, H.,Gohlke, C., and Oliphant, T. E. Array programmingwith NumPy. Nature, 585:357–362, 2020. doi: 10.1038/s41586-020-2649-2.
* Hendrycks, D. and Gimpel, K. Gaussian error linear units(gelus). arXiv preprint arXiv:1606.08415, 2016.
* Hendrycks, D., Liu, X., Wallace, E., Dziedzic, A., Krishnan,R., and Song, D. Pretrained transformers improve out-of￾distribution robustness. arXiv preprint arXiv:2004.06100,2020.
* Hernandez, F., Nguyen, V., Ghannay, S., Tomashenko, N. A.,and Est`eve, Y. Ted-lium 3: twice as much data and corpusrepartition for experiments on speaker adaptation. InSPECOM, 2018.
* Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K.,Salakhutdinov, R., and Mohamed, A. Hubert: Self￾supervised speech representation learning by maskedprediction of hidden units. IEEE/ACM Transactions onAudio, Speech, and Language Processing, 29:3451–3460,2021a.
* Hsu, W.-N., Sriram, A., Baevski, A., Likhomanenko, T.,Xu, Q., Pratap, V., Kahn, J., Lee, A., Collobert, R., Syn￾naeve, G., et al. Robust wav2vec 2.0: Analyzing do￾main shift in self-supervised pre-training. arXiv preprintarXiv:2104.01027, 2021b.
* Huang, G., Sun, Y., Liu, Z., Sedra, D., and Weinberger,K. Q. Deep networks with stochastic depth. In Europeanconference on computer vision, pp. 646–661. Springer,2016.
* Jia, R. and Liang, P. Adversarial examples for evalu￾ating reading comprehension systems. arXiv preprintarXiv:1707.07328, 2017.
* Johnson, M., Schuster, M., Le, Q. V., Krikun, M., Wu, Y.,Chen, Z., Thorat, N., Vi´egas, F., Wattenberg, M., Corrado,G., et al. Google’s multilingual neural machine translationsystem: Enabling zero-shot translation. Transactions ofthe Association for Computational Linguistics, 5:339–351, 2017.
* Kendall, T. and Farrington, C. The corpus of regionalafrican american language. Version 2021.07. Eugene, OR:The Online Resources for African American LanguageProject. http://oraal.uoregon.edu/coraal,2021. Accessed: 2022-09-01.
* Koenecke, A., Nam, A., Lake, E., Nudell, J., Quartey, M.,Mengesha, Z., Toups, C., Rickford, J. R., Jurafsky, D.,and Goel, S. Racial disparities in automated speech recog￾nition. Proceedings of the National Academy of Sciences,117(14):7684–7689, 2020.
* Kolesnikov, A., Beyer, L., Zhai, X., Puigcerver, J., Yung,J., Gelly, S., and Houlsby, N. Big transfer (bit): Generalvisual representation learning. In European conferenceon computer vision, pp. 491–507. Springer, 2020.
* Kuchaiev, O., Li, J., Nguyen, H., Hrinchuk, O., Leary, R.,Ginsburg, B., Kriman, S., Beliaev, S., Lavrukhin, V.,Cook, J., et al. Nemo: a toolkit for building ai applicationsusing neural modules. arXiv preprint arXiv:1909.09577,2019.
* Lake, B. M., Ullman, T. D., Tenenbaum, J. B., and Gersh￾man, S. J. Building machines that learn and think likepeople. Behavioral and brain sciences, 40, 2017.
* Liao, H., McDermott, E., and Senior, A. Large scale deepneural network acoustic modeling with semi-supervisedtraining data for youtube video transcription. In 2013IEEE Workshop on Automatic Speech Recognition andUnderstanding, pp. 368–373. IEEE, 2013.
* Likhomanenko, T., Xu, Q., Pratap, V., Tomasello, P., Kahn,J., Avidov, G., Collobert, R., and Synnaeve, G. Rethink￾ing evaluation in asr: Are our models robust enough?arXiv preprint arXiv:2010.11745, 2020.
* Loshchilov, I. and Hutter, F. Decoupled weight decay regu￾larization. arXiv preprint arXiv:1711.05101, 2017.
* Luong, M.-T., Le, Q. V., Sutskever, I., Vinyals, O., andKaiser, L. Multi-task sequence to sequence learning. arXiv preprint arXiv:1511.06114, 2015.
* Mahajan, D., Girshick, R., Ramanathan, V., He, K., Paluri,M., Li, Y., Bharambe, A., and Van Der Maaten, L. Ex￾ploring the limits of weakly supervised pretraining. InProceedings of the European conference on computervision (ECCV), pp. 181–196, 2018.
* Mauch, M. and Ewert, S. The audio degradation toolbox andits application to robustness evaluation. In Proceedings ofthe 14th International Society for Music Information Re￾trieval Conference (ISMIR 2013), Curitiba, Brazil, 2013.
* accepted.
* McCann, B., Keskar, N. S., Xiong, C., and Socher, R. Thenatural language decathlon: Multitask learning as ques￾tion answering. arXiv preprint arXiv:1806.08730, 2018.
* Meyer, J., Rauchenstein, L., Eisenberg, J. D., and Howell,N. Artie bias corpus: An open dataset for detecting de￾mographic bias in speech applications. In Proceedings ofthe 12th Language Resources and Evaluation Conference,pp. 6462–6468, Marseille, France, May 2020. EuropeanLanguage Resources Association. ISBN 979-10-95546-34-4. URL https://aclanthology.org/2020.
* lrec-1.796.
* Miller, J., Krauth, K., Recht, B., and Schmidt, L. The effectof natural distribution shift on question answering models.
* In ICML, 2020.
* Mohamed, A.-r., Dahl, G., Hinton, G., et al. Deep belief net￾works for phone recognition. In Nips workshop on deeplearning for speech recognition and related applications,volume 1, pp. 39, 2009.
* Narayanan, A., Misra, A., Sim, K. C., Pundak, G., Tripathi,A., Elfeky, M., Haghani, P., Strohman, T., and Bacchi￾ani, M. Toward domain-invariant speech recognition vialarge scale training. In 2018 IEEE Spoken LanguageTechnology Workshop (SLT), pp. 441–447. IEEE, 2018.
* Panayotov, V., Chen, G., Povey, D., and Khudanpur, S.
* Librispeech: an asr corpus based on public domain au￾dio books. In 2015 IEEE international conference onacoustics, speech and signal processing (ICASSP), pp.
* 5206–5210. IEEE, 2015.
* pandas development team, T. pandas-dev/pandas: Pan￾das, February 2020. URL https://doi.org/10.
* 5281/zenodo.3509134.
* Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B.,Cubuk, E. D., and Le, Q. V. SpecAugment: A simple dataaugmentation method for automatic speech recognition. arXiv preprint arXiv:1904.08779, 2019.
* Pascanu, R., Mikolov, T., and Bengio, Y. On the difficultyof training recurrent neural networks. In Internationalconference on machine learning, pp. 1310–1318. PMLR,2013.
* Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J.,Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga,L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison,M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L.,Bai, J., and Chintala, S. Pytorch: An imperative style,high-performance deep learning library. In Advancesin Neural Information Processing Systems 32, pp. 8024–8035, 2019.
* Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V.,Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P.,Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cour￾napeau, D., Brucher, M., Perrot, M., and Duchesnay, E.
* Scikit-learn: Machine learning in Python. Journal ofMachine Learning Research, 12:2825–2830, 2011.
* Polyak, B. T. and Juditsky, A. B. Acceleration of stochasticapproximation by averaging. SIAM journal on controland optimization, 30(4):838–855, 1992.
* Pratap, V., Sriram, A., Tomasello, P., Hannun, A. Y.,Liptchinsky, V., Synnaeve, G., and Collobert, R. Mas￾sively multilingual asr: 50 languages, 1 model, 1 billionparameters. ArXiv, abs/2007.03001, 2020a.
* Pratap, V., Xu, Q., Sriram, A., Synnaeve, G., and Collobert,R. Mls: A large-scale multilingual dataset for speechresearch. arXiv preprint arXiv:2012.03411, 2020b.
* Press, O. and Wolf, L. Using the output embedding toimprove language models. In Proceedings of the 15thConference of the European Chapter of the Associa￾tion for Computational Linguistics: Volume 2, ShortPapers, pp. 157–163, Valencia, Spain, April 2017. As￾sociation for Computational Linguistics. URL https://aclanthology.org/E17-2025.
* Provilkov, I., Emelianenko, D., and Voita, E. Bpe-dropout:Simple and effective subword regularization. arXivpreprint arXiv:1910.13267, 2019.
* Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., andSutskever, I. Language models are unsupervised multitasklearners. 2019.
* Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G.,Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark,J., Krueger, G., and Sutskever, I. Learning transferablevisual models from natural language supervision. arXivpreprint arXiv:2103.00020, 2021.
* Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S.,Matena, M., Zhou, Y., Li, W., Liu, P. J., et al. Exploringthe limits of transfer learning with a unified text-to-texttransformer. J. Mach. Learn. Res., 21(140):1–67, 2020.
* Ravanelli, M., Parcollet, T., Plantinga, P., Rouhe, A., Cor￾nell, S., Lugosch, L., Subakan, C., Dawalatabad, N.,Heba, A., Zhong, J., Chou, J.-C., Yeh, S.-L., Fu, S.-W.,Liao, C.-F., Rastorgueva, E., Grondin, F., Aris, W., Na,H., Gao, Y., Mori, R. D., and Bengio, Y. SpeechBrain: Ageneral-purpose speech toolkit, 2021. arXiv:2106.04624.
* Recht, B., Roelofs, R., Schmidt, L., and Shankar, V.
* Do ImageNet classifiers generalize to ImageNet? InChaudhuri, K. and Salakhutdinov, R. (eds.), Proceed￾ings of the 36th International Conference on MachineLearning, volume 97 of Proceedings of Machine Learn￾ing Research, pp. 5389–5400. PMLR, 09–15 Jun 2019.
* URL https://proceedings.mlr.press/v97/recht19a.html.
* Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S.,Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein,M., et al. Imagenet large scale visual recognition chal￾lenge. International journal of computer vision, 115(3):211–252, 2015.
* Schultz, T. and Kirchhoff, K. Multilingual speech process￾ing. Elsevier, 2006.
* Seide, F., Li, G., Chen, X., and Yu, D. Feature engineeringin context-dependent deep neural networks for conver￾sational speech transcription. In 2011 IEEE Workshopon Automatic Speech Recognition & Understanding, pp.
* 24–29. IEEE, 2011.
* Sennrich, R., Haddow, B., and Birch, A. Neural machinetranslation of rare words with subword units. arXivpreprint arXiv:1508.07909, 2015.
* Speer, R. ftfy. Zenodo, 2019. URL https://doi.org/10.5281/zenodo.2591652. Version 5.5.
* Sutskever, I., Vinyals, O., and Le, Q. V. Sequence to se￾quence learning with neural networks. Advances in neuralinformation processing systems, 27, 2014.
* Taori, R., Dave, A., Shankar, V., Carlini, N., Recht, B.,and Schmidt, L. Measuring robustness to naturaldistribution shifts in image classification. In Larochelle,H., Ranzato, M., Hadsell, R., Balcan, M., and Lin,H. (eds.), Advances in Neural Information ProcessingSystems, volume 33, pp. 18583–18599. Curran Asso￾ciates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/d8330f857a17c53d217014ee776bfd50-Paper.pdf.
* Torralba, A. and Efros, A. A. Unbiased look at dataset bias. CVPR 2011, pp. 1521–1528, 2011.
* Toshniwal, S., Sainath, T. N., Weiss, R. J., Li, B., Moreno,P. J., Weinstein, E., and Rao, K. Multilingual speechrecognition with a single end-to-end model. 2018 IEEEInternational Conference on Acoustics, Speech and Sig￾nal Processing (ICASSP), pp. 4904–4908, 2018.
* Valk, J. and Alum¨ae, T. Voxlingua107: a dataset for spokenlanguage recognition. In 2021 IEEE Spoken LanguageTechnology Workshop (SLT), pp. 652–658. IEEE, 2021.
* Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Atten￾tion is all you need. In Advances in neural informationprocessing systems, pp. 5998–6008, 2017.
* Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M.,Reddy, T., Cournapeau, D., Burovski, E., Peterson, P.,Weckesser, W., Bright, J., van der Walt, S. J., Brett, M.,Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J.,Jones, E., Kern, R., Larson, E., Carey, C. J., Polat, ˙I.,Feng, Y., Moore, E. W., VanderPlas, J., Laxalde, D.,Perktold, J., Cimrman, R., Henriksen, I., Quintero, E. A.,Harris, C. R., Archibald, A. M., Ribeiro, A. H., Pedregosa,F., van Mulbregt, P., and SciPy 1.0 Contributors. SciPy1.0: Fundamental Algorithms for Scientific Computingin Python. Nature Methods, 17:261–272, 2020. doi:10.1038/s41592-019-0686-2.
* Wang, C., Tang, Y., Ma, X., Wu, A., Okhonko, D., and Pino,J. fairseq s2t: Fast speech-to-text modeling with fairseq. arXiv preprint arXiv:2010.05171, 2020a.
* Wang, C., Wu, A., and Pino, J. Covost 2 and massivelymultilingual speech-to-text translation. arXiv preprintarXiv:2007.10310, 2020b.
* Wang, C., Riviere, M., Lee, A., Wu, A., Talnikar, C., Haziza,D., Williamson, M., Pino, J., and Dupoux, E. Voxpopuli:A large-scale multilingual speech corpus for representa￾tion learning, semi-supervised learning and interpretation. arXiv preprint arXiv:2101.00390, 2021.
* Wang, P., Sainath, T. N., and Weiss, R. J. Multitask trainingwith text data for end-to-end speech recognition. arXivpreprint arXiv:2010.14318, 2020c.
* Watanabe, S., Mandel, M., Barker, J., Vincent, E., Arora,A., Chang, X., Khudanpur, S., Manohar, V., Povey, D.,Raj, D., et al. Chime-6 challenge: Tackling multispeakerspeech recognition for unsegmented recordings. arXivpreprint arXiv:2004.09249, 2020.
* Xu, Q., Baevski, A., Likhomanenko, T., Tomasello, P., Con￾neau, A., Collobert, R., Synnaeve, G., and Auli, M. Self￾training and pre-training are complementary for speechrecognition. In ICASSP 2021-2021 IEEE InternationalConference on Acoustics, Speech and Signal Processing(ICASSP), pp. 3030–3034. IEEE, 2021.
* Zhang, Y., Qin, J., Park, D. S., Han, W., Chiu, C.-C., Pang,R., Le, Q. V., and Wu, Y. Pushing the limits of semi￾supervised learning for automatic speech recognition. arXiv preprint arXiv:2010.10504, 2020.
* Zhang, Y., Park, D. S., Han, W., Qin, J., Gulati, A., Shor, J.,Jansen, A., Xu, Y., Huang, Y., Wang, S., et al. BigSSL:Exploring the frontier of large-scale semi-supervisedlearning for automatic speech recognition. arXiv preprintarXiv:2109.13226, 2021.
* Robust Speech Recognition via Large-Scale Weak Supervision 


## A. Evaluation Datasets.
### A.1. Short-form English-only datasets
* LibriSpeech (Panayotov et al., 2015): We used the test-clean and test-other splits from the LibriSpeech ASR corpus. * TED-LIUM 3 (Hernandez et al., 2018): We used the test split of TED-LIUM Release 3, using the segmented manual transcripts included in the release.
* Common Voice 5.1 (Ardila et al., 2019): We downloaded the English subset of Common Voice Corpus 5.1 from the official website. * Artie bias corpus (Meyer et al., 2020): We used the Artie bias corpus. This is a subset of the Common Voice dataset.
* CallHome and Switchboard: We used the two corpora from LDC2002S09 and LDC2002T43. * WSJ: We used LDC93S6B and LDC94S13B and followed the s5 recipe to preprocess the dataset.
* CORAAL: We used the 231 interviews from CORAAL (Kendall & Farrington, 2021) and used the preprocessing script from the FairSpeech project. * CHiME-6: For CHiME-6 (Watanabe et al., 2020), we downloaded the CHiME-5 dataset and followed the stage 0 of the s5 track1 recipe to create the CHiME-6 dataset which fixes synchronization. We then used the binaural recordings (* P??.wav) and the corresponding transcripts.
* AMI-IHM and AMI-SDM1: We preprocessed the AMI Corpus by following the stage 0 ad 2 of the s5b recipe.

### A.2. Long-form English-only datasets
* TED-LIUM 3 (Hernandez et al., 2018): We used the 11 full-length TED talks from the test split of TED-LIUM Release 3, slicing the source audio files between the beginning of the first labeled segment and the end of the last labeled segment of each talk, and we used the concatenated text as the label.
* Meanwhile: This dataset consists of 64 segments from The Late Show with Stephen Colbert. The YouTube video ID and the corresponding start and end timestamps are available as part of the code release. The labels are collected from the closed-caption data for each video and corrected with manual inspection.
* Rev16: We use a subset of 16 files from the 30 podcast episodes in Rev.AI’s Podcast Transcription Benchmark, after finding that there are multiple cases where a significant portion of the audio and the labels did not match, mostly on the parts introducing the sponsors. We selected 16 episodes that do not have this error, whose “file number”s are: 3 4 9 10 11 14 17 18 20 21 23 24 26 27 29 32
* Kincaid46: This dataset consists of 46 audio files and the corresponding transcripts compiled in the blog article ¡Which automatic transcription service is the most accurate - 2018¿ by Jason Kincaid. We used the 46 audio files and reference transcripts from the Airtable widget in the article. For the human transcription benchmark in the paper, we use a subset of 25 examples from this data, whose “Ref ID”s are: 2 4 5 8 9 10 12 13 14 16 19 21 23 25 26 28 29 30 33 35 36 37 42 43 45
* Earnings-21 (Del Rio et al., 2021) and Earnings-22: We used the files available in the speech-datasets repository, as of their 202206 version.
* CORAAL: We used the 231 full-length interviews and transcripts from (Kendall & Farrington, 2021).



### A.3. Multilingual datasets
* Multilingual LibriSpeech (Pratap et al., 2020b): We used the test splits from each language in the Multilingual LibriSpeech (MLS) corpus. * Fleurs (Conneau et al., 2022): We collected audio files and transcripts using the implementation available as HuggingFace datasets. To use as a translation dataset, we matched the numerical utterance IDs to find the corresponding transcript in English.
* VoxPopuli (Wang et al., 2021): We used the get asr data.py script from the official repository to collect the ASR data in 16 languages, including English.
* Common Voice 9 (Ardila et al., 2019): We downloaded the Common Voice Corpus 9 from the official website. * CoVOST 2 (Wang et al., 2020b): We collected the X into English data collected using the official repository.

## B. Compared Models
For comparison, we use the following models from HuggingFace, downloaded as of September 2022 using version 4.21.0 of the transformers library:
* facebook/wav2vec2-large-960h-lv60-self (Xu et al., 2021) * facebook/wav2vec2-large-robust-ft-libri-960h (Hsu et al., 2021b) * facebook/wav2vec2-base-100h (Baevski et al., 2020) * facebook/wav2vec2-base-960h (Baevski et al., 2020) * facebook/wav2vec2-large-960h (Baevski et al., 2020) * facebook/hubert-large-ls960-ft (Hsu et al., 2021a) * facebook/hubert-xlarge-ls960-ft (Hsu et al., 2021a) * facebook/s2t-medium-librispeech-asr (Wang et al., 2020a) * facebook/s2t-large-librispeech-asr (Wang et al., 2020a) * microsoft/unispeech-sat-base-100h-libri-ft (Chen et al., 2022a) * nvidia/stt en conformer ctc large (Kuchaiev et al., 2019) * nvidia/stt en conformer transducer xlarge (Kuchaiev et al., 2019) * speechbrain/asr-crdnn-rnnlm-librispeech (Ravanelli et al., 2021) * speechbrain/asr-transformer-transformerlm-librispeech (Ravanelli et al., 2021)

We note that all of the models above are entirely or partly trained on LibriSpeech.

## C. Text Standardization
Since Whisper may output any UTF-8 string rather than a restricted set of graphemes, the rules for text standardization need to be more intricate and comprehensive than those defined on e.g. ASCII characters. We perform the following steps to normalize English texts in different styles into a standardized form, which is a best-effort attempt to penalize only when a word error is caused by actually mistranscribing a word, and not by formatting or punctuation differences.

1. Remove any phrases between matching brackets ([, ]).
2. Remove any phrases between matching parentheses ((,)).
3. Remove any of the following words: hmm, mm, mhm, mmm, uh, um
4. Remove whitespace characters that comes before an apostrophe ’
5. Convert standard or informal contracted forms of English into the original form.
6. Remove commas (,) between digits
7. Remove periods (.) not followed by numbers
8. Remove symbols as well as diacritics from the text, where symbols are the characters with the Unicode category starting with M, S, or P, except period, percent, and currency symbols that may be detected in the next step.
9. Detect any numeric expressions of numbers and currencies and replace with a form using Arabic numbers, e.g. “Ten thousand dollars” → “$10000”.
10. Convert British spellings into American spellings.
11. Remove remaining symbols that are not part of any numeric expressions.
12. Replace any successive whitespace characters with a space.

A different, language-specific set of transformations would be needed to equivalently normalize non-English text, but due to our lack of linguistic knowledge to build such normalizers for all languages, we resort to the following basic standardization for non-English text:
1. Remove any phrases between matching brackets ([, ]).
2. Remove any phrases between matching parentheses ((,)).
3. Replace any markers, symbols, and punctuation characters with a space, i.e. when the Unicode category of each character in the NFKC-normalized string starts with M, S, or P.
4. make the text lowercase.
5. replace any successive whitespace characters with a space.
Additionally, we put a space between every letter for the languages that do not use spaces to separate words, namely Chinese, Japanese, Thai, Lao, and Burmese, effectively measuring the character error rate instead.

We note that the above is an imperfect solution, and it will sometimes produce unintended and unexpected outputs. We do not claim that the text format resulting from the above is more “correct” in any measure. Rather, the procedures above are designed to better distinguish between innocuous differences in wording and genuine mistranscriptions. Python code for the standardization procedures above is available as part of our code and model release to facilitate future iterations and improvements on text standardization.



## D. Raw Performance Tables
### D.1. English Transcription
#### D.1.1. GREEDY DECODING
Table 8. English transcription WER (%) with greedy decoding

####  D.1.2. BEAM SEARCH WITH TEMPERATURE FALLBACK
Table 9. English transcription WER (%) with beam search and temperature fallback LibriSpeech.test-clean LibriSpeech.test-other TED-LIUM3 WSJ CallHome Switchboard CommonVoice5.1 Artie CORAAL CHiME6 AMI-IHM

### D.2. Multilingual Transcription
#### D.2.1. MULTILINGUAL LIBRISPEECH
#### D.2.2. COMMON VOICE 9
Table 11. WER (%) on CommonVoice9

#### D.2.3. VOXPOPULI
Table 12. WER (%) on VoxPopuli Dutch English French German Italian


#### D.2.4. FLEURS
Table 13. WER (%) on Fleurs

### D.3. Speech Translation
#### D.3.1. FLEURS
Table 14. BLEU scores on Fleurs

#### D.3.2. COVOST 2
Table 15. BLEU scores on CoVoST2

### D.4. Long-form Transcription
Table 16. Long-form English transcription WER (%)

## E. Training Dataset Statistics
Chinese 23446 65% English Speech Recognition (438,218 hours) 18% Translation (125,739 hours) 17% Multilingual Speech Recognition (117,113 hours)<br/>
Figure 11. Training dataset statistics

## F. Hyperparameters
Table 17. Whisper training hyperparameters.
 
Table 18. Hyperparameters changed for Whisper Large V2.

Table 19. Whisper model learning rates.

