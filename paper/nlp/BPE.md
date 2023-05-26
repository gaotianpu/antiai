# Neural Machine Translation of Rare Words with Subword Units
稀有词具有子词单元的的神经机器翻译 2015.8.31 https://arxiv.org/abs/1508.07909

## 阅读笔记
* BPE(Byte Pair Encoding), 3.2 Byte Pair Encoding (BPE) 节，
* from gpt-2: We observed BPE including many versions of common words like dog since they occur in many variations such as dog. dog! dog? . This results in a sub-optimal allocation of limited vocabulary slots and model capacity. To avoid this, we prevent BPE from merging across character categories for any byte sequence. We add an exception for spaces which significantly improves the compression efficiency while adding only minimal fragmentation of words across multiple vocab tokens.
我们观察到 BPE 包括许多版本的常用词，例如dog，因为它们出现在许多变体中，例如: dog. dog! dog? . 这导致有限词汇槽和模型容量的次优分配。 为避免这种情况，我们阻止 BPE 合并任何字节序列的跨字符类别。 我们为空格添加了一个例外，这显著提高了压缩效率，同时仅在多个词汇标记中添加了最少的单词碎片。
* gpt-2: https://github.com/openai/gpt-2/blob/master/src/encoder.py
 

## Abstract
Neural machine translation (NMT) models typically operate with a fixed vocabulary, but translation is an open-vocabulary problem. Previous work addresses the translation of out-of-vocabulary words by backing off to a dictionary. In this paper, we introduce a simpler and more effective approach, making the NMT model capable of open-vocabulary translation by encoding rare and unknown words as sequences of subword units. This is based on the intuition that various word classes are translatable via smaller units than words, for instance names (via character copying or transliteration), compounds (via compositional translation), and cognates and loanwords (via phonological and morphological transformations). We discuss the suitability of different word segmentation techniques, including simple character ngram models and a segmentation based on the byte pair encoding compression algorithm, and empirically show that subword models improve over a back-off dictionary baseline for the WMT 15 translation tasks English→German and English→Russian by up to 1.1 and 1.3 BLEU, respectively. 

神经机器翻译 (NMT) 模型通常使用固定词汇进行操作，但翻译是一个开放式词汇问题。 以前的工作通过退回到字典来解决词汇外单词的翻译问题。 在本文中，我们介绍了一种更简单、更有效的方法，通过将稀有和未知词编码为子词单元序列，使 NMT 模型能够进行开放式词汇翻译。 这是基于这样一种直觉，即各种词类可以通过比词更小的单位进行翻译，例如名称(通过字符复制或音译)、复合词(通过组合翻译)以及同源词和外来词(通过语音和形态转换)。 我们讨论了不同分词技术的适用性，包括简单的字符 ngram 模型和基于字节对编码(BPE)压缩算法的分词，并根据经验表明子词模型在 WMT 15 翻译任务的英语→德语翻译任务中改进了退避词典基线 和英语→俄语分别高达 1.1 和 1.3 BLEU。

## 1 Introduction
Neural machine translation has recently shown impressive results (Kalchbrenner and Blunsom, 2013; Sutskever et al., 2014; Bahdanau et al., 2015). However, the translation of rare words is an open problem. The vocabulary of neural models is typically limited to 30 000–50 000 words, but translation is an open-vocabulary problem, and especially for languages with productive word formation processes such as agglutination and compounding, translation models require mechanisms that go below the word level. As an example, consider compounds such as the German Abwasser|behandlungs|anlange ‘sewage water treatment plant’, for which a segmented, variable-length representation is intuitively more appealing than encoding the word as a fixed-length vector.

神经机器翻译最近显示出令人印象深刻的结果(Kalchbrenner 和 Blunsom，2013 ;Sutskever et al., 2014 ;Bahdanau et al., 2015)。 然而，稀有词的翻译是一个悬而未决的问题。 神经模型的词汇量通常限制在 30000~50000 个单词，但翻译是一个开放式词汇问题，特别是对于具有生产性单词形成过程(如凝集和复合)的语言，翻译模型需要低于单词级别的机制 . 例如，考虑诸如德语 Abwasser|behandlungs|anlange ‘sewage water treatment plant’ 之类的合成词，对于这种合成词，分段的可变长度表示在直觉上比将单词编码为固定长度的向量更具吸引力。

For word-level NMT models, the translation of out-of-vocabulary words has been addressed through a back-off to a dictionary look-up (Jean et al., 2015; Luong et al., 2015b). We note that such techniques make assumptions that often do not hold true in practice. For instance, there is not always a 1-to-1 correspondence between source and target words because of variance in the degree of morphological synthesis between languages, like in our introductory compounding example. Also, word-level models are unable to translate or generate unseen words. Copying unknown words into the target text, as done by (Jean et al., 2015; Luong et al., 2015b), is a reasonable strategy for names, but morphological changes and transliteration is often required, especially if alphabets differ.

对于词级 NMT 模型，词汇外单词的翻译已通过回退到字典查找来解决(Jean et al., 2015; Luong et al., 2015b)。 我们注意到，此类技术做出的假设在实践中通常不成立。 例如，由于语言之间形态合成程度的差异，源词和目标词之间并不总是一一对应，就像我们介绍的复合样本中那样。 此外，词级模型无法翻译或生成看不见的词。 将未知词复制到目标文本中，如 (Jean et al., 2015; Luong et al., 2015b) 所做的那样，是一种合理的名称策略，但通常需要形态变化和音译，尤其是在字母表不同的情况下。

We investigate NMT models that operate on the level of subword units. Our main goal is to model open-vocabulary translation in the NMT network itself, without requiring a back-off model for rare words. In addition to making the translation process simpler, we also find that the subword models achieve better accuracy for the translation of rare words than large-vocabulary models and back-off dictionaries, and are able to productively generate new words that were not seen at training time. Our analysis shows that the neural networks are able to learn compounding and transliteration from subword representations.

我们研究了在子词单元级别上运行的 NMT 模型。 我们的主要目标是在 NMT 网络本身中对开放式词汇翻译进行建模，而不需要针对稀有词的回退模型。 除了使翻译过程更简单外，我们还发现子词模型在翻译稀有词方面比大词汇量模型和回退词典具有更高的准确性，并且能够有效地生成训练中未见过的新词 时间。 我们的分析表明，神经网络能够从子词表示中学习复合和音译。

This paper has two main contributions:
* We show that open-vocabulary neural machine translation is possible by encoding (rare) words via subword units. We find our architecture simpler and more effective than using large vocabularies and back-off dictionaries (Jean et al., 2015; Luong et al., 2015b).
* We adapt byte pair encoding (BPE) (Gage, 1994), a compression algorithm, to the task of word segmentation. BPE allows for the representation of an open vocabulary through a fixed-size vocabulary of variable-length character sequences, making it a very suitable word segmentation strategy for neural network models. 

这篇论文有两个主要贡献：
* 我们表明，通过子词单元对(稀有)词进行编码，可以实现开放式词汇神经机器翻译。 我们发现我们的架构比使用大型词汇表和回退词典更简单、更有效(Jean et al., 2015 ;Luong et al., 2015b)。
* 我们采用称为 字节对编码 (BPE)(Gage，1994)的压缩算法用于分词任务。 BPE 允许通过可变长度字符序列的固定大小词汇表来表示开放词汇表，使其成为一种非常适合神经网络模型的分词策略。

## 2 Neural Machine Translation
We follow the neural machine translation architecture by Bahdanau et al. (2015), which we will briefly summarize here. However, we note that our approach is not specific to this architecture.

我们遵循 Bahdanau et al. (2015) 的神经机器翻译架构，我们将在这里简要总结一下。 然而，我们注意到我们的方法并不特定于此架构。

The neural machine translation system is implemented as an encoder-decoder network with recurrent neural networks.

神经机器翻译系统被实现为具有递归神经网络的编码器-解码器网络。

The encoder is a bidirectional neural network with gated recurrent units (Cho et al., 2014) that reads an input sequence $x = (x_1, ..., x_m)$ and calculates a forward sequence of hidden states $(−→h_1, ..., −→h_m)$, and a backward sequence $( ←−h_1, ..., ←−h_m)$. The hidden states $−→h_j$ and $←−h_j$ are concatenated to obtain the annotation vector $h_j$ .

编码器是一个双向神经网络，带有门控循环单元 (Cho et al., 2014)，它读取输入序列 $x = (x_1, ..., x_m)$ 并计算隐藏状态的正向序列 $(−→h_1 , ..., −→h_m)$，以及一个后向序列 $( ←−h_1, ..., ←−h_m)$。 连接隐藏状态 $−→h_j$ 和 $←−h_j$ 以获得注释向量 $h_j$ 。

The decoder is a recurrent neural network that predicts a target sequence $y = (y_1, ..., y_n)$. Each word $y_i$ is predicted based on a recurrent hidden state $s_i$ , the previously predicted word $y_{i−1}$, and a context vector $c_i$. $c_i$ is computed as a weighted sum of the annotations $h_j$ . The weight of each annotation $h_j$ is computed through an alignment model $α_{ij}$ , which models the probability that $y_i$ is aligned to $x_j$ . The alignment model is a singlelayer feedforward neural network that is learned jointly with the rest of the network through backpropagation.

解码器是一个递归神经网络，它预测目标序列 $y = (y_1, ..., y_n)$。 每个词 $y_i$ 都是基于循环隐藏状态 $s_i$ 、先前预测的词 $y_{i−1}$ 和上下文向量 $c_i$ 预测的。 $c_i$ 计算为注释 $h_j$ 的加权和。 每个注释 $h_j$ 的权重是通过对齐模型 $α_{ij}$ 计算的，该模型模拟 $y_i$ 与 $x_j$ 对齐的概率。 对齐模型是一个单层前馈神经网络，通过反向传播与网络的其余部分共同学习。

A detailed description can be found in (Bahdanau et al., 2015). Training is performed on a parallel corpus with stochastic gradient descent. For translation, a beam search with small beam size is employed. 

详细描述可以在 (Bahdanau et al., 2015) 中找到。 训练是在具有随机梯度下降的平行语料库上进行的。 对于翻译，采用小波束尺寸的波束搜索。

## 3 Subword Translation
The main motivation behind this paper is that the translation of some words is transparent in that they are translatable by a competent translator even if they are novel to him or her, based on a translation of known subword units such as morphemes or phonemes. Word categories whose translation is potentially transparent include:
* named entities. Between languages that share an alphabet, names can often be copied from source to target text. Transcription or transliteration may be required, especially if the alphabets or syllabaries differ. Example: Barack Obama (English; German) Барак Обама (Russian) バラク・オバマ (ba-ra-ku o-ba-ma) (Japanese)
* cognates and loanwords. Cognates and loanwords with a common origin can differ in regular ways between languages, so that character-level translation rules are sufficient (Tiedemann, 2012). Example: claustrophobia (English) Klaustrophobie (German)Клаустрофобия (Klaustrofobiâ) (Russian)
* morphologically complex words. Words containing multiple morphemes, for instance formed via compounding, affixation, or inflection, may be translatable by translating the morphemes separately. Example: solar system (English) Sonnensystem (Sonne + System) (German)Naprendszer (Nap + Rendszer) (Hungarian)

这篇论文背后的主要动机是，一些词的翻译是透明的，因为它们可以由有能力的翻译者翻译，即使它们对他或她来说是新的，基于已知子词单元(如语素或音素)的翻译。 翻译可能透明的词类包括：
* 命名实体。 在共享字母表的语言之间，名称通常可以从源文本复制到目标文本。 可能需要转录或音译，尤其是当字母表或音节表不同时。 样本：Barack Obama (English; German) Барак Обама (Russian) バラク・オバマ (ba-ra-ku o-ba-ma) (Japanese)
* 同源词和外来词。 具有共同来源的同源词和外来词在不同语言之间可能有规律地不同，因此字符级别的翻译规则就足够了(Tiedemann，2012)。 样本：claustrophobia (English) Klaustrophobie (German)Клаустрофобия (Klaustrofobiâ) (Russian)
* 形态复杂的词。 包含多个词素的词，例如通过复合、词缀或词形变化形成的词，可以通过单独翻译词素来翻译。 样本：solar system (English) Sonnensystem (Sonne + System) (German)Naprendszer (Nap + Rendszer) (Hungarian)

In an analysis of 100 rare tokens (not among the 50 000 most frequent types) in our German training data(1Primarily parliamentary proceedings and web crawl data.) , the majority of tokens are potentially translatable from English through smaller units. We find 56 compounds, 21 names, 6 loanwords with a common origin (emancipate→emanzipieren), 5 cases of transparent affixation (sweetish ‘sweet’ + ‘-ish’ → süßlich ‘süß’ + ‘-lich’), 1 number and 1 computer language identifier.

在对我们的德语训练数据(1主要是议会程序和网络抓取数据。)中的 100 个稀有标记(不在 50 000 个最常见类型中)的分析中，大多数标记都可能通过较小的单元从英语翻译。 我们发现 56 个复合词、21 个名称、6 个具有共同起源的外来词 (emancipate→emanzipieren)、5 个透明词缀案例 (sweetish 'sweet' + '-ish' → süßlich 'süß' + '-lich')、1 个数字和 1 计算机语言标识符。

Our hypothesis is that a segmentation of rare words into appropriate subword units is sufficient to allow for the neural translation network to learn transparent translations, and to generalize this knowledge to translate and produce unseen words(2Not every segmentation we produce is transparent. While we expect no performance benefit from opaque segmentations, i.e. segmentations where the units cannot be translated independently, our NMT models show robustness towards oversplitting). We provide empirical support for this hypothesis in Sections 4 and 5. First, we discuss different subword representations.

我们的假设是，将稀有词分割成适当的子词单元足以让神经翻译网络学习透明翻译，并将这种知识泛化以翻译和生成看不见的词(2并非我们生成的每个分割都是透明的。虽然我们期望 不透明分割没有性能优势，即单元不能独立翻译的分割，我们的 NMT 模型显示出对过度分割的稳健性)。 我们在第 4 节和第 5 节中为该假设提供实证支持。首先，我们讨论不同的子词表示。

### 3.1 Related Work
For Statistical Machine Translation (SMT), the translation of unknown words has been the subject of intensive research.

对于统计机器翻译(SMT)，未知词的翻译一直是深入研究的主题。

A large proportion of unknown words are names, which can just be copied into the target text if both languages share an alphabet. If alphabets differ, transliteration is required (Durrani et al., 2014). Character-based translation has also been investigated with phrase-based models, which proved especially successful for closely related languages (Vilar et al., 2007; Tiedemann, 2009; Neubig et al., 2012).

很大一部分未知词是名称，如果两种语言共享一个字母表，则可以将其复制到目标文本中。 如果字母不同，则需要音译(Durrani et al., 2014)。 基于字符的翻译也用基于短语的模型进行了研究，事实证明这对于密切相关的语言特别成功(Vilar et al., 2007 ;Tiedemann，2009 ;Neubig et al., 2012)。

The segmentation of morphologically complex words such as compounds is widely used for SMT, and various algorithms for morpheme segmentation have been investigated (Nießen and Ney, 2000; Koehn and Knight, 2003; Virpioja et al., 2007; Stallard et al., 2012). Segmentation algorithms commonly used for phrase-based SMT tend to be conservative in their splitting decisions, whereas we aim for an aggressive segmentation that allows for open-vocabulary translation with a compact network vocabulary, and without having to resort to back-off dictionaries.

复合词等形态复杂词的分割广泛用于 SMT，并且研究了各种词素分割算法(Nießen 和 Ney，2000 ;Koehn 和 Knight，2003 ;Virpioja et al., 2007 ;Stallard et al., 2012) ). 通常用于基于短语的 SMT 的分割算法在其拆分决策中往往是保守的，而我们的目标是进行积极的分割，允许使用紧凑的网络词汇进行开放式词汇翻译，而不必诉诸回退词典。

The best choice of subword units may be taskspecific. For speech recognition, phone-level language models have been used (Bazzi and Glass, 2000). Mikolov et al. (2012) investigate subword language models, and propose to use syllables. For multilingual segmentation tasks, multilingual algorithms have been proposed (Snyder and Barzilay, 2008). We find these intriguing, but inapplicable at test time.

子词单元的最佳选择可能是特定于任务的。 对于语音识别，使用了音素级语言模型(Bazzi and Glass, 2000). Mikolov et al. (2012) 调查子词语言模型，并建议使用音节。 对于多语言分割任务，已经提出了多语言算法(Snyder 和 Barzilay，2008)。 我们发现这些很有趣，但在测试时不适用。

Various techniques have been proposed to produce fixed-length continuous word vectors based on characters or morphemes (Luong et al., 2013; Botha and Blunsom, 2014; Ling et al., 2015a; Kim et al., 2015). An effort to apply such techniques to NMT, parallel to ours, has found no significant improvement over word-based approaches (Ling et al., 2015b). One technical difference from our work is that the attention mechanism still operates on the level of words in the model by Ling et al. (2015b), and that the representation of each word is fixed-length. We expect that the attention mechanism benefits from our variable-length representation: the network can learn to place attention on different subword units at each step. Recall our introductory example Abwasserbehandlungsanlange, for which a subword segmentation avoids the information bottleneck of a fixed-length representation.

已经提出了各种技术来生成基于字符或词素的固定长度的连续词向量(Luong et al., 2013 ;Botha 和 Blunsom，2014 ;Ling et al., 2015a; Kim et al., 2015)。 与我们类似的将此类技术应用于 NMT 的努力并未发现比基于单词的方法有显著改进(Ling et al., 2015b)。 与我们工作的一个技术差异是，注意力机制仍然在 Ling et al 的模型中的单词级别上运行。 (2015b)，并且每个单词的表示都是固定长度的。 我们期望注意力机制受益于我们的可变长度表示：网络可以学习在每一步将注意力放在不同的子词单元上。 回想一下我们的介绍性样本 Abwasserbehandlungsanlange，其中子词分割避免了固定长度表示的信息瓶颈。

Neural machine translation differs from phrasebased methods in that there are strong incentives to minimize the vocabulary size of neural models to increase time and space efficiency, and to allow for translation without back-off models. At the same time, we also want a compact representation of the text itself, since an increase in text length reduces efficiency and increases the distances over which neural models need to pass information.

神经机器翻译不同于基于短语的方法，因为有强烈的动机最小化神经模型的词汇量以提高时间和空间效率，并允许在没有回退模型的情况下进行翻译。 同时，我们还想要文本本身的紧凑表示，因为文本长度的增加会降低效率并增加神经模型需要传递信息的距离。

A simple method to manipulate the trade-off between vocabulary size and text size is to use shortlists of unsegmented words, using subword units only for rare words. As an alternative, we propose a segmentation algorithm based on byte pair encoding (BPE), which lets us learn a vocabulary that provides a good compression rate of the text.

在词汇量和文本大小之间进行权衡的一种简单方法是使用未分段词的候选列表，仅对稀有词使用子词单元。 作为替代方案，我们提出了一种基于字节对编码 (BPE) 的分段算法，它让我们可以学习提供良好文本压缩率的词汇表。

### 3.2 Byte Pair Encoding (BPE)
Byte Pair Encoding (BPE) (Gage, 1994) is a simple data compression technique that iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte. We adapt this algorithm for word segmentation. Instead of merging frequent pairs of bytes, we merge characters or character sequences.

字节对编码 (BPE)(Gage，1994)是一种简单的数据压缩技术，它用一个未使用的字节迭代地替换序列中最常见的字节对。 我们将此算法用于分词。 我们不是合并高频字节对，而是合并字符或字符序列。

Firstly, we initialize the symbol vocabulary with the character vocabulary, and represent each word as a sequence of characters, plus a special end-of-word symbol ‘·’, which allows us to restore the original tokenization after translation. We iteratively count all symbol pairs and replace each occurrence of the most frequent pair (‘A’, ‘B’) with a new symbol ‘AB’. Each merge operation produces a new symbol which represents a character n-gram. Frequent character n-grams (or whole words) are eventually merged into a single symbol, thus BPE requires no shortlist. The final symbol vocabulary size is equal to the size of the initial vocabulary, plus the number of merge operations – the latter is the only hyperparameter of the algorithm.

首先，我们用字符词汇表初始化符号词汇表，并将每个单词表示为一个字符序列，加上一个特殊的词尾符号“·”，这使我们能够在翻译后恢复原始标记化。 我们迭代地计算所有符号对，并用新符号“AB”替换每次出现的最高频对(“A”、“B”)。 每个合并操作都会产生一个新符号，代表一个字符 n-gram。 频繁出现的字符 n-gram(或整个单词)最终会合并为一个符号，因此 BPE 不需要入围名单。 最终的符号词汇表大小等于初始词汇表的大小加上合并操作次数 —— 后者是算法的唯一超参数。

<!--合并操作次数，ab替换 a,b，是否还要保留？-->

For efficiency, we do not consider pairs that cross word boundaries. The algorithm can thus be run on the dictionary extracted from a text, with each word being weighted by its frequency. A minimal Python implementation is shown in Algorithm 1. In practice, we increase efficiency by indexing all pairs, and updating data structures incrementally.

为了效率，我们不考虑跨越单词边界的对。 因此，该算法可以在从文本中提取的字典上运行，每个单词都按其频率加权。 算法 1 中显示了一个最小的 Python 实现。在实践中，我们通过索引所有对并逐步更新数据结构来提高效率。
<!--中文按单个汉字？-->

The main difference to other compression algorithms, such as Huffman encoding, which have been proposed to produce a variable-length encoding of words for NMT (Chitnis and DeNero, 2015), is that our symbol sequences are still interpretable as subword units, and that the network can generalize to translate and produce new words (unseen at training time) on the basis of these subword units.

与其他压缩算法(例如霍夫曼编码)的主要区别在于，已提出为 NMT 生成可变长度的单词编码(Chitnis 和 DeNero，2015)，我们的符号序列仍然可以解释为子词单元，并且 网络可以在这些子词单元的基础上泛化翻译和生成新词(在训练时看不到)。

```python
# Algorithm 1 Learn BPE operations

import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,
'n e w e s t </w>':6, 'w i d e s t </w>':3}
num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)

```

Figure 1: BPE merge operations learned from dictionary {‘low’, ‘lowest’, ‘newer’, ‘wider’}. 
图 1：从字典 {‘low’、‘lowest’、‘newer’、‘wider’} 中学习的 BPE 合并操作。

Figure 1 shows a toy example of learned BPE operations. At test time, we first split words into sequences of characters, then apply the learned operations to merge the characters into larger, known symbols. This is applicable to any word, and allows for open-vocabulary networks with fixed symbol vocabularies(3 The only symbols that will be unknown at test time are unknown characters, or symbols of which all occurrences in the training text have been merged into larger symbols, like ‘safeguar’, which has all occurrences in our training text merged into ‘safeguard’. We observed no such symbols at test time, but the issue could be easily solved by recursively reversing specific merges until all symbols are known). In our example, the OOV ‘lower’ would be segmented into ‘low er·’. 

图1 显示了学习 BPE 操作的玩具样本。 在测试时，我们首先将单词拆分为字符序列，然后应用学习的操作将字符合并为更大的已知符号。 这适用于任何单词，并允许具有固定符号词汇表的开放式词汇网络(3 在测试时唯一未知的符号是未知字符，或者训练文本中所有出现的符号都已合并为更大的符号 ，比如“safeguar”，它在我们的训练文本中出现的所有地方都合并到“safeguard”中。我们在测试时没有观察到这样的符号，但是这个问题可以很容易地通过递归反转特定的合并来解决，直到所有的符号都已知)。 在我们的样本中，OOV“lower”将被分割为“lower·”。

We evaluate two methods of applying BPE: learning two independent encodings, one for the source, one for the target vocabulary, or learning the encoding on the union of the two vocabularies (which we call joint BPE)(4In practice, we simply concatenate the source and target side of the training set to learn joint BPE). The former has the advantage of being more compact in terms of text and vocabulary size, and having stronger guarantees that each subword unit has been seen in the training text of the respective language, whereas the latter improves consistency between the source and the target segmentation. If we apply BPE independently, the same name may be segmented differently in the two languages, which makes it harder for the neural models to learn a mapping between the subword units. To increase the consistency between English and Russian segmentation despite the differing alphabets, we transliterate the Russian vocabulary into Latin characters with ISO-9 to learn the joint BPE encoding, then transliterate the BPE merge operations back into Cyrillic to apply them to the Russian training text(5 Since the Russian training text also contains words that use the Latin alphabet, we also apply the Latin BPE operations. ). 

我们评估了两种应用 BPE 的方法：学习两种独立的编码，一种用于源词汇，一种用于目标词汇，或者学习两种词汇联合的编码(我们称之为联合 BPE)(4 在实践中，我们简单地将 训练集的源端和目标端来学习联合 BPE)。 前者的优点是在文本和词汇量方面更紧凑，并且更有力地保证每个子词单元都已在各自语言的训练文本中看到，而后者则提高了源和目标分割之间的一致性。 如果我们独立应用 BPE，相同的名称在两种语言中可能会有不同的分割，这使得神经模型更难学习子词单元之间的映射。 尽管字母表不同，但为了提高英语和俄语分割之间的一致性，我们使用 ISO-9 将俄语词汇音译为拉丁字符以学习联合 BPE 编码，然后将 BPE 合并操作音译回西里尔字母以将其应用于俄语训练文本 (5 由于俄语训练文本也包含使用拉丁字母的单词，我们也应用拉丁 BPE 操作。)。

## 4 Evaluation
We aim to answer the following empirical questions:
* Can we improve the translation of rare and unseen words in neural machine translation by representing them via subword units?
* Which segmentation into subword units performs best in terms of vocabulary size, text size, and translation quality?

我们旨在回答以下实证问题：
* 我们能否通过子词单元表示它们来改进神经机器翻译中罕见和不可见词的翻译？
* 哪种子词单元分割在词汇量、文本大小和翻译质量方面表现最好？

We perform experiments on data from the shared translation task of WMT 2015. For English→German, our training set consists of 4.2 million sentence pairs, or approximately 100 million tokens. For English→Russian, the training set consists of 2.6 million sentence pairs, or approximately 50 million tokens. We tokenize and truecase the data with the scripts provided in Moses (Koehn et al., 2007). We use newstest2013 as development set, and report results on newstest2014 and newstest2015.

我们对来自 WMT 2015 共享翻译任务的数据进行实验。对于英语→德语，我们的训练集包含 420 万个句子对，或大约 1 亿个标记。 对于英语→俄语，训练集包含 260 万个句子对，或大约 5000 万个标记。 我们使用 Moses (Koehn et al., 2007) 中提供的脚本对数据进行标记化和 true case。 我们使用newstest2013作为开发集，并在newstest2014和newstest2015上报告结果。

We report results with BLEU (mteval-v13a.pl), and CHRF3 (Popovi´c, 2015), a character n-gram F3 score which was found to correlate well with human judgments, especially for translations out of English (Stanojevi´c et al., 2015). Since our main claim is concerned with the translation of rare and unseen words, we report separate statistics for these. We measure these through unigram F1, which we calculate as the harmonic mean of clipped unigram precision and recall.(6Clipped unigram precision is essentially 1-gram BLEU without brevity penalty. )

我们使用 BLEU (mteval-v13a.pl) 和 CHRF3 (Popovi´c, 2015) 报告结果，这是一种字符 n-gram F3 分数，被发现与人类判断密切相关，尤其是对于非英语的翻译 (Stanojevi´c et al., 2015)。 由于我们的主要主张与罕见和未见过的单词的翻译有关，因此我们报告了这些单独的统计数据。 我们通过 unigram F1 来衡量这些，我们将其计算为裁剪的 unigram 精度和召回率的调和平均值。(6Clipped unigram 精度本质上是 1-gram BLEU，没有简洁惩罚。)

We perform all experiments with Groundhog(7 github.com/sebastien-j/LV_groundhog) (Bahdanau et al., 2015). We generally follow settings by previous work (Bahdanau et al., 2015; Jean et al., 2015). All networks have a hidden layer size of 1000, and an embedding layer size of 620. Following Jean et al. (2015), we only keep a shortlist of τ = 30000 words in memory.

我们使用 Groundhog(7 github.com/sebastien-j/LV_groundhog) 进行所有实验(Bahdanau et al., 2015)。 我们通常遵循以前工作的设置(Bahdanau et al., 2015 ;Jean et al., 2015)。 所有网络的隐藏层大小为 1000，嵌入层大小为 620。继 Jean et al 之后。 (2015)，我们只在内存中保留了 τ = 30000 个单词的候选列表。

During training, we use Adadelta (Zeiler, 2012), a minibatch size of 80, and reshuffle the training set between epochs. We train a network for approximately 7 days, then take the last 4 saved models (models being saved every 12 hours), and continue training each with a fixed embedding layer (as suggested by (Jean et al., 2015)) for 12 hours. We perform two independent training runs for each models, once with cut-off for gradient clipping (Pascanu et al., 2013) of 5.0, once with a cut-off of 1.0 – the latter produced better single models for most settings. We report results of the system that performed best on our development set (newstest2013), and of an ensemble of all 8 models.

在训练期间，我们使用 Adadelta (Zeiler, 2012)，小批量大小为 80，并在不同时期之间重新洗牌训练集。 我们训练一个网络大约 7 天，然后使用最后 4 个保存的模型(模型每 12 小时保存一次)，并使用固定的嵌入层继续训练每个模型(如 (Jean et al., 2015) 所建议的那样)12 小时 . 我们对每个模型执行两次独立的训练运行，一次使用 5.0 的梯度裁剪截止值 (Pascanu et al., 2013)，一次使用 1.0 的截止值——后者为大多数设置生成了更好的单一模型。 我们报告了在我们的开发集 (newstest2013) 上表现最好的系统的结果，以及所有 8 个模型的集合的结果。

We use a beam size of 12 for beam search, with probabilities normalized by sentence length.We use a bilingual dictionary based on fast-align (Dyer et al., 2013). For our baseline, this serves as back-off dictionary for rare words. We also use the dictionary to speed up translation for all experiments, only performing the softmax over a filtered list of candidate translations (like Jean et al. (2015), we use K = 30000; K′ = 10).

我们使用 12 的光束大小进行光束搜索，概率按句子长度归一化。我们使用基于快速对齐的双语词典 (Dyer et al., 2013)。 对于我们的基线，这用作稀有词的回退字典。 我们还使用字典来加速所有实验的翻译，只对过滤后的候选翻译列表执行 softmax(如 Jean et al (2015)，我们使用 K = 30000; K' = 10)。

### 4.1 Subword statistics 子词统计
Apart from translation quality, which we will verify empirically, our main objective is to represent an open vocabulary through a compact fixed-size subword vocabulary, and allow for efficient training and decoding.(8The time complexity of encoder-decoder architectures is at least linear to sequence length, and oversplitting harms efficiency.)

除了我们将根据经验验证的翻译质量外，我们的主要目标是通过紧凑的固定大小的子词词汇表来表示开放词汇表，并允许进行有效的训练和解码。(8编码器-解码器架构的时间复杂度至少是线性的 序列长度，过度分裂会损害效率。)

Statistics for different segmentations of the German side of the parallel data are shown in Table 1. A simple baseline is the segmentation of words into character n-grams.(9Our character n-grams do not cross word boundaries. We mark whether a subword is word-final or not with a special character, which allows us to restore the original tokenization. ) Character n-grams allow for different trade-offs between sequence length (# tokens) and vocabulary size (# types), depending on the choice of n. The increase in sequence length is substantial; one way to reduce sequence length is to leave a shortlist of the k most frequent word types unsegmented. Only the unigram representation is truly open-vocabulary. However, the unigram representation performed poorly in preliminary experiments, and we report translation results with a bigram representation, which is empirically better, but unable to produce some tokens in the test set with the training set vocabulary.

表1 显示了并行数据的德语侧不同分割的统计数据。一个简单的基线是将单词分割成字符 n-grams。9 字符 n-grams 允许在序列长度(# tokens)之间进行不同的权衡 和词汇量(# 类型)，取决于 n 的选择。 序列长度的增加是实质性的;  减少序列长度的一种方法是保留 k 个最常见词类型的候选列表，不进行分段。 只有 unigram 表示才是真正的开放式词汇表。 然而，unigram 表示在初步实验中表现不佳，我们用 bigram 表示报告翻译结果，这在经验上更好，但无法在测试集中使用训练集词汇生成一些标记。

Table 1: Corpus statistics for German training corpus with different word segmentation techniques. #UNK: number of unknown tokens in newstest2013. △: (Koehn and Knight, 2003); *: (Creutz and Lagus, 2002); ⋄: (Liang, 1983).
表 1：使用不同分词技术的德语训练语料库的语料库统计。 #UNK：newstest2013 中未知标记的数量。 △：(科恩和奈特，2003);  *：(Creutz 和 Lagus，2002);  ⋄: (Liang, 1983).

We report statistics for several word segmentation techniques that have proven useful in previous SMT research, including frequency-based compound splitting (Koehn and Knight, 2003), rulebased hyphenation (Liang, 1983), and Morfessor (Creutz and Lagus, 2002). We find that they only moderately reduce vocabulary size, and do not solve the unknown word problem, and we thus find them unsuitable for our goal of open-vocabulary translation without back-off dictionary.

我们报告了几种在以前的 SMT 研究中被证明有用的分词技术的统计数据，包括基于频率的复合拆分(Koehn 和 Knight，2003)、基于规则的连字符(Liang，1983)和 Morfessor(Creutz 和 Lagus，2002)。 我们发现它们只是适度地减少了词汇量，并没有解决未知词问题，因此我们发现它们不适合我们在没有回退词典的情况下进行开放式词汇翻译的目标。

BPE meets our goal of being open-vocabulary, and the learned merge operations can be applied to the test set to obtain a segmentation with no unknown symbols.(10 Joint BPE can produce segments that are unknown because they only occur in the English training text, but these are rare (0.05% of test tokens).) Its main difference from the character-level model is that the more compact representation of BPE allows for shorter sequences, and that the attention model operates on variable-length units.(11We highlighted the limitations of word-level attention in section 3.1. At the other end of the spectrum, the character level is suboptimal for alignment (Tiedemann, 2009). ) Table 1 shows BPE with 59 500 merge operations, and joint BPE with 89 500 operations.

BPE 满足了我们开放词汇的目标，学习到的合并操作可以应用于测试集以获得没有未知符号的分段。(10 Joint BPE 可以产生未知的分段，因为它们只出现在英语训练文本中， 但这些很少见(占测试令牌的 0.05%)。它与字符级模型的主要区别在于 BPE 的更紧凑表示允许更短的序列，并且注意力模型在可变长度单元上运行。(11我们强调 3.1 节中词级注意力的局限性。在频谱的另一端，字符级对于对齐不是最优的 (Tiedemann, 2009)。)表 1 显示了具有 59 500 个合并操作的 BPE 和具有 89 500 个操作的联合 BPE .

In practice, we did not include infrequent subword units in the NMT network vocabulary, since there is noise in the subword symbol sets, e.g. because of characters from foreign alphabets. Hence, our network vocabularies in Table 2 are typically slightly smaller than the number of types in Table 1. 

在实践中，我们没有在 NMT 网络词汇表中包含不常见的子词单元，因为子词符号集中存在噪声，例如 因为来自外国字母的字符。 因此，我们在表 2 中的网络词汇表通常略小于表 1 中的类型数量。

Table 2: English→German translation performance (BLEU, CHRF3 and unigram F1) on newstest2015. Ens-8: ensemble of 8 models. Best NMT system in bold. Unigram F1 (with ensembles) is computed for all words (n = 44085), rare words (not among top 50 000 in training set; n = 2900), and OOVs (not in training set; n = 1168). 
表 2：newstest2015 上的英语→德语翻译性能(BLEU、CHRF3 和 unigram F1)。 Ens-8：8 个模型的集合。 最佳 NMT 系统以粗体显示。 为所有单词 (n = 44085)、稀有单词(不在训练集中的前 50000 个中; n = 2900)和 OOV(不在训练集中; n = 1168)计算 Unigram F1(带集成)。


### 4.2 Translation experiments
English→German translation results are shown in Table 2; English→Russian results in Table 3.

英→德翻译结果如表2所示;  英语→俄语结果见表3。

Our baseline WDict is a word-level model with a back-off dictionary. It differs from WUnk in that the latter uses no back-off dictionary, and just represents out-of-vocabulary words as UNK (12We use UNK for words that are outside the model vocabulary, and OOV for those that do not occur in the training text.). The back-off dictionary improves unigram F1 for rare and unseen words, although the improvement is smaller for English→Russian, since the back-off dictionary is incapable of transliterating names.

我们的基线 WDict 是一个带有回退字典的词级模型。 与WUnk不同的是后者不使用back-off字典，只是将out-of-vocabulary的词表示为UNK(12We use UNK for words that are outside of the model vocabulary, OOV表示那些在训练中没有出现的词 文本。)。 退避词典改进了罕见和未见过的单词的 unigram F1，尽管英语→俄语的改进较小，因为退避词典无法音译名称。

All subword systems operate without a back-off dictionary. We first focus on unigram F1, where all systems improve over the baseline, especially for rare words (36.8%→41.8% for EN→DE;  26.5%→29.7% for EN→RU). For OOVs, the baseline strategy of copying unknown words works well for English→German. However, when alphabets differ, like in English→Russian, the subword models do much better. 

所有子词系统都在没有退避字典的情况下运行。 我们首先关注 unigram F1，其中所有系统都比基线有所改进，尤其是对于稀有词(EN→DE 为 36.8%→41.8%; EN→RU 为 26.5%→29.7%)。 对于 OOV，复制未知词的基线策略适用于英语→德语。 然而，当字母表不同时，比如英语→俄语，子词模型会做得更好。

Unigram F1 scores indicate that learning the BPE symbols on the vocabulary union (BPEJ90k) is more effective than learning them separately (BPE-60k), and more effective than using character bigrams with a shortlist of 50 000 unsegmented words (C2-50k), but all reported subword segmentations are viable choices and outperform the back-off dictionary baseline.

Unigram F1 分数表明，在词汇联合 (BPEJ90k) 上学习 BPE 符号比单独学习它们 (BPE-60k) 更有效，并且比使用包含 50 000 个未分段单词的候选列表 (C2-50k) 的字符二元组更有效， 但是所有报告的子词分割都是可行的选择，并且优于回退字典基线。

Our subword representations cause big improvements in the translation of rare and unseen words, but these only constitute 9-11% of the test sets. Since rare words tend to carry central information in a sentence, we suspect that BLEU and CHRF3 underestimate their effect on translation quality. Still, we also see improvements over the baseline in total unigram F1, as well as BLEU and CHRF3, and the subword ensembles outperform the WDict baseline by 0.3–1.3 BLEU and  0.6–2 CHRF3. There is some inconsistency between BLEU and CHRF3, which we attribute to the fact that BLEU has a precision bias, and CHRF3 a recall bias.

我们的子词表示在罕见和未见过的词的翻译方面有很大的改进，但这些只占测试集的 9-11%。 由于稀有词往往在句子中携带中心信息，我们怀疑 BLEU 和 CHRF3 低估了它们对翻译质量的影响。 尽管如此，我们还看到总 unigram F1 以及 BLEU 和 CHRF3 的基线有所改进，子词组合的性能优于 WDict 基线 0.3-1.3 BLEU 和 0.6-2 CHRF3。 BLEU 和 CHRF3 之间存在一些不一致，我们将其归因于 BLEU 具有精确偏差，而 CHRF3 具有召回偏差。

For English→German, we observe the best BLEU score of 25.3 with C2-50k, but the best CHRF3 score of 54.1 with BPE-J90k. For comparison to the (to our knowledge) best non-neural MT system on this data set, we report syntaxbased SMT results (Sennrich and Haddow, 2015).We observe that our best systems outperform the syntax-based system in terms of BLEU, but not in terms of CHRF3. Regarding other neural systems, Luong et al. (2015a) report a BLEU score of  25.9 on newstest2015, but we note that they use an ensemble of 8 independently trained models, and also report strong improvements from applying dropout, which we did not use. We are confident that our improvements to the translation of rare words are orthogonal to improvements achievable through other improvements in the network architecture, training algorithm, or better ensembles.

对于英语→德语，我们观察到 C2-50k 的最佳 BLEU 得分为 25.3，但 BPE-J90k 的最佳 CHRF3 得分为 54.1。 为了与该数据集上的(据我们所知)最佳非神经 MT 系统进行比较，我们报告了基于句法的 SMT 结果(Sennrich 和 Haddow，2015)。我们观察到我们的最佳系统在 BLEU 方面优于基于句法的系统， 但不是就 CHRF3 而言。 关于其他神经系统，Luong et al. (2015a) 在 newstest2015 上报告了 25.9 的 BLEU 分数，但我们注意到他们使用了 8 个独立训练模型的集合，并且还报告了我们没有使用的 dropout 带来的巨大改进。 我们相信，我们对稀有词翻译的改进与通过网络架构、训练算法或更好的集成等其他改进实现的改进是正交的。

For English→Russian, the state of the art is the phrase-based system by Haddow et al. (2015). It outperforms our WDict baseline by 1.5 BLEU. The subword models are a step towards closing this gap, and BPE-J90k yields an improvement of 1.3 BLEU, and 2.0 CHRF3, over WDict.

对于英语→俄语，最先进的是 Haddow et al 的基于短语的系统。 (2015)。 它比我们的 WDict 基线高出 1.5 BLEU。 子词模型是朝着缩小这一差距迈出的一步，BPE-J90k 比 WDict 提高了 1.3 BLEU 和 2.0 CHRF3。

As a further comment on our translation results, we want to emphasize that performance variability is still an open problem with NMT. On our development set, we observe differences of up to 1 BLEU between different models. For single systems, we report the results of the model that performs best on dev (out of 8), which has a stabilizing effect, but how to control for randomness deserves further attention in future research. 

作为对我们翻译结果的进一步评论，我们想强调性能可变性仍然是 NMT 的一个未解决问题。 在我们的开发集上，我们观察到不同模型之间最多 1 BLEU 的差异。 对于单个系统，我们报告了在 dev 上表现最好的模型的结果(满分 8 个)，它具有稳定效果，但如何控制随机性值得在未来的研究中进一步关注。

## 5 Analysis
### 5.1 Unigram accuracy
Our main claims are that the translation of rare and unknown words is poor in word-level NMT models, and that subword models improve the translation of these word types. To further illustrate the effect of different subword segmentations on the translation of rare and unseen words, we plot target-side words sorted by their frequency in the training set.(13We perform binning of words with the same training set frequency, and apply bezier smoothing to the graph. ) To analyze the effect of vocabulary size, we also include the system C2-3/500k, which is a system with the same vocabulary size as the WDict baseline, and character bigrams to represent unseen words.

我们的主要主张是，在词级 NMT 模型中，稀有词和未知词的翻译效果很差，而子词模型可以改善这些词类型的翻译。 为了进一步说明不同子词分割对稀有词和未见词翻译的影响，我们绘制了按训练集中频率排序的目标侧词。(13我们对具有相同训练集频率的词进行分箱，并应用贝塞尔平滑 到图表。)为了分析词汇量的影响，我们还包括系统 C2-3/500k，这是一个与 WDict 基线具有相同词汇量的系统，以及表示未见过单词的字符二元组。

Figure 2 shows results for the English–German ensemble systems on newstest2015. Unigram F1 of all systems tends to decrease for lowerfrequency words. The baseline system has a spike in F1 for OOVs, i.e. words that do not occur in the training text. This is because a high proportion of OOVs are names, for which a copy from the source to the target text is a good strategy for English→German.

图 2 显示了英德合奏系统在 newstest2015 上的结果。 对于低频词，所有系统的 Unigram F1 都趋于降低。 基线系统在 OOV 的 F1 中有一个尖峰，即没有出现在训练文本中的单词。 这是因为很大比例的 OOV 是名称，从源文本复制到目标文本是英语→德语的一个很好的策略。

The systems with a target vocabulary of 500 000 words mostly differ in how well they translate words with rank > 500 000. A back-off dictionary is an obvious improvement over producing UNK, but the subword system C2-3/500k achieves better performance. Note that all OOVs that the backoff dictionary produces are words that are copied from the source, usually names, while the subword systems can productively form new words such as compounds.

目标词汇量为 500 000 个单词的系统在翻译排名 > 500 000 的单词时的表现大不相同。回退词典比生成 UNK 有明显的改进，但子词系统 C2-3/500k 实现了更好的性能。 请注意，退避词典生成的所有 OOV 都是从源中复制的词，通常是名称，而子词系统可以有效地形成新词，例如复合词。

For the 50 000 most frequent words, the representation is the same for all neural networks, and all neural networks achieve comparable unigram F1 for this category. For the interval between frequency rank 50 000 and 500 000, the comparison between C2-3/500k and C2-50k unveils an interesting difference. The two systems only differ in the size of the shortlist, with C2-3/500k representing words in this interval as single units, and C250k via subword units. We find that the performance of C2-3/500k degrades heavily up to frequency rank 500 000, at which point the model switches to a subword representation and performance recovers. The performance of C2-50k remains more stable. We attribute this to the fact that subword units are less sparse than words. In our training set, the frequency rank 50 000 corresponds to a frequency of 60 in the training data; the frequency rank 500 000 to a frequency of 2. Because subword representations are less sparse, reducing the size of the network vocabulary, and representing more words via subword units, can lead to better performance.

对于 50 000 个最频繁出现的单词，所有神经网络的表示都是相同的，并且所有神经网络都实现了该类别的可比一元组 F1。 对于频率等级 50 000 和 500 000 之间的间隔，C2-3/500k 和 C2-50k 之间的比较揭示了一个有趣的差异。 这两个系统仅在入围列表的大小上有所不同，C2-3/500k 将此间隔中的单词表示为单个单元，而 C250k 通过子单词单元表示。 我们发现 C2-3/500k 的性能严重下降到频率等级 500 000，此时模型切换到子词表示并且性能恢复。 C2-50k的表现更加稳定。 我们将此归因于子词单元不如单词稀疏的事实。 在我们的训练集中，频率排名 50 000 对应于训练数据中的频率 60;  频率等级从 500 000 到频率 2。因为子词表示不那么稀疏，减少网络词汇表的大小，并通过子词单元表示更多的词，可以带来更好的性能。

The F1 numbers hide some qualitative differences between systems. For English→German, WDict produces few OOVs (26.5% recall), but with high precision (60.6%) , whereas the subword systems achieve higher recall, but lower precision. We note that the character bigram model C2-50k produces the most OOV words, and achieves relatively low precision of 29.1% for this category. However, it outperforms the back-off dictionary in recall (33.0%). BPE-60k, which suffers from transliteration (or copy) errors due to segmentation inconsistencies, obtains a slightly better precision (32.4%), but a worse recall (26.6%). In contrast to BPE-60k, the joint BPE encoding of BPEJ90k improves both precision (38.6%) and recall (29.8%).

F1 数字隐藏了系统之间的一些质的差异。 对于英语→德语，WDict 产生的 OOV 很少(召回率 26.5%)，但准确率很高(60.6%)，而子词系统的召回率较高，但准确率较低。 我们注意到字符二元模型 C2-50k 产生了最多的 OOV 词，并且该类别的精度相对较低，为 29.1%。 然而，它在召回率 (33.0%) 方面优于回退字典。 BPE-60k 由于分段不一致而遭受音译(或复制)错误，其精度略高(32.4%)，但召回率较差(26.6%)。 与 BPE-60k 相比，BPEJ90k 的联合 BPE 编码提高了精度 (38.6%) 和召回率 (29.8%)。

For English→Russian, unknown names can only rarely be copied, and usually require transliteration. Consequently, the WDict baseline performs more poorly for OOVs (9.2% precision; 5.2% recall), and the subword models improve both precision and recall (21.9% precision and 15.6% recall for BPE-J90k). The full unigram F1 plot is shown in Figure 3. 

对于英语→俄语，未知的名字只能很少复制，通常需要音译。 因此，WDict 基线对 OOV 的表现更差(9.2% 的精度; 5.2% 的召回率)，而子词模型提高了精度和召回率(BPE-J90k 的精度和召回率分别为 21.9% 和 15.6%)。 完整的 unigram F1 图如图 3 所示。

Table 3: English→Russian translation performance (BLEU, CHRF3 and unigram F1) on newstest2015. Ens-8: ensemble of 8 models. Best NMT system in bold. Unigram F1 (with ensembles) is computed for all words (n = 55654), rare words (not among top 50 000 in training set; n = 5442), and OOVs (not in training set; n = 851). 
表 3：英语→俄语翻译性能(BLEU、CHRF3 和 unigram F1)在 newstest2015 上的表现。 Ens-8：8 个模型的集合。 最佳 NMT 系统以粗体显示。 为所有单词 (n = 55654)、稀有单词(不在训练集中的前 50000 个中; n = 5442)和 OOV(不在训练集中; n = 851)计算 Unigram F1(带集成)。

Figure 2: English→German unigram F1 on newstest2015 plotted by training set frequency rank for different NMT systems. 
图 2：根据不同 NMT 系统的训练集频率等级绘制的 newstest2015 上的英语→德语 unigram F1。

Figure 3: English→Russian unigram F1 on newstest2015 plotted by training set frequency rank for different NMT systems.
图 3：newstest2015 上英语→俄语 unigram F1 按不同 NMT 系统的训练集频率排名绘制。

### 5.2 Manual Analysis
Table 4 shows two translation examples for the translation direction English→German, Table 5 for English→Russian. The baseline system fails for all of the examples, either by deleting content (health), or by copying source words that should be translated or transliterated. The subword translations of health research institutes show that the subword systems are capable of learning translations when oversplitting (research→Fo|rs|ch|un|g), or when the segmentation does not match morpheme boundaries: the segmentation Forschungs|instituten would be linguistically more plausible, and simpler to align to the English research institutes, than the segmentation Forsch|ungsinstitu|ten in the BPE-60k system, but still, a correct translation is produced. If the systems have failed to learn a translation due to data sparseness, like for asinine, which should be translated as dumm, we see translations that are wrong, but could be plausible for (partial) loanwords (asinine Situation→Asinin-Situation).

表 4 显示了翻译方向为英语→德语的两个翻译样本，表 5 为英语→俄语。 基线系统对于所有样本都失败了，要么删除内容(健康)，要么复制应该翻译或音译的源词。 健康研究机构的子词翻译表明，子词系统能够在过度分割(研究→Fo|rs|ch|un|g)时学习翻译，或者当分割与词素边界不匹配时：分割 Forschungs|instituten 将是 在语言上比 BPE-60k 系统中的分割 Forsch|ungsinsitu|ten 更合理，更容易与英语研究机构保持一致，但仍然产生了正确的翻译。 如果系统由于数据稀疏而无法学习翻译，比如 asinine，它应该被翻译成 dumm，我们会看到错误的翻译，但对于(部分)借词(asinine Situation→Asinin-Situation)来说可能是合理的。

The English→Russian examples show that the subword systems are capable of transliteration. However, transliteration errors do occur, either due to ambiguous transliterations, or because of non-consistent segmentations between source and target text which make it hard for the system to learn a transliteration mapping. Note that the BPE-60k system encodes Mirzayeva inconsistently for the two language pairs (Mirz|ayeva→Мир|за|ева Mir|za|eva). This example is still translated correctly, but we observe spurious insertions and deletions of characters in the BPE-60k system. An example is the transliteration of rakfisk, where a п is inserted and a к is deleted. We trace this error back to translation pairs in the training data with inconsistent segmentations, such as (p|rak|ri|ti→пра|крит|и (pra|krit|i)), from which the translation (rak→пра) is erroneously learned. The segmentation of the joint BPE system (BPE-J90k) is more consistent (pra|krit|i→пра|крит|и (pra|krit|i)). 

英语→俄语的例子表明子词系统是可以音译的。 然而，音译错误确实会发生，要么是由于音译含糊不清，要么是因为源文本和目标文本之间的分割不一致，这使得系统很难学习音译映射。 请注意，BPE-60k 系统对两个语言对 (Mirz|ayeva→Мир|за|ева Mir|za|eva) 的 Mirzayeva 编码不一致。 这个例子仍然被正确翻译，但我们观察到 BPE-60k 系统中的字符的虚假插入和删除。 一个例子是 rakfisk 的音译，其中插入了 п 并删除了 к。 我们将此错误追溯到训练数据中具有不一致分段的翻译对，例如 (p|rak|ri|ti→пра|крит|и (pra|krit|i))，从中翻译 (rak→пра) 被错误地学习了。 联合 BPE 系统 (BPE-J90k) 的分割更加一致 (pra|krit|i→пра|крит|и (pra|krit|i))。
 

Table 4: English→German translation example. “|” marks subword boundaries. 
表 4：英语→德语翻译样本。 “|” 标记子词边界。
 
Table 5: English→Russian translation examples. “|” marks subword boundaries. 
表 5：英语→俄语翻译样本。 “|” 标记子词边界。



## 6 Conclusion
The main contribution of this paper is that we show that neural machine translation systems are capable of open-vocabulary translation by representing rare and unseen words as a sequence of subword units.(14The source code of the segmentation algorithms is available at https://github.com/rsennrich/subword-nmt. ) This is both simpler and more effective than using a back-off translation model. We introduce a variant of byte pair encoding for word segmentation, which is capable of encoding open vocabularies with a compact symbol vocabulary of variable-length subword units. We show performance gains over the baseline with both BPE segmentation, and a simple character bigram segmentation. 

本文的主要贡献是，我们表明神经机器翻译系统能够通过将罕见和未见过的词表示为子词单元序列来进行开放式词汇翻译。(14 分割算法的源代码可在 https:// github.com/rsennrich/subword-nmt。)这比使用回退翻译模型更简单、更有效。 我们引入了一种用于分词的字节对编码变体，它能够使用可变长度子词单元的紧凑符号词汇表对开放词汇表进行编码。 我们通过 BPE 分割和简单的字符二元分割展示了相对于基线的性能提升。

Our analysis shows that not only out-ofvocabulary words, but also rare in-vocabulary words are translated poorly by our baseline NMT system, and that reducing the vocabulary size of subword models can actually improve performance. In this work, our choice of vocabulary size is somewhat arbitrary, and mainly motivated by comparison to prior work. One avenue of future research is to learn the optimal vocabulary size for a translation task, which we expect to depend on the language pair and amount of training data, automatically. We also believe there is further potential in bilingually informed segmentation algorithms to create more alignable subword units, although the segmentation algorithm cannot rely on the target text at runtime.

我们的分析表明，我们的基线 NMT 系统不仅翻译了词汇外的单词，而且罕见的词汇内的单词也翻译得很差，而且减少子词模型的词汇量实际上可以提高性能。 在这项工作中，我们对词汇量大小的选择有些随意，主要是为了与之前的工作进行比较。 未来研究的一个途径是学习翻译任务的最佳词汇量，我们希望这取决于语言对和训练数据量，自动。 我们还相信双语信息分割算法有进一步的潜力来创建更多可对齐的子词单元，尽管分割算法不能在运行时依赖目标文本。

While the relative effectiveness will depend on language-specific factors such as vocabulary size, we believe that subword segmentations are suitable for most language pairs, eliminating the need for large NMT vocabularies or back-off models.

虽然相对有效性将取决于特定语言的因素，例如词汇量大小，但我们认为子词分割适用于大多数语言对，从而消除了对大型 NMT 词汇表或退避模型的需求。

## Acknowledgments
We thank Maja Popovi´c for her implementation of CHRF, with which we verified our reimplementation. The research presented in this publication was conducted in cooperation with Samsung Electronics Polska sp. z o.o. Samsung R&D Institute Poland. This project received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement 645452 (QT21).

## References
* Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural Machine Translation by Jointly Learning to Align and Translate. In Proceedings of the International Conference on Learning Representations (ICLR) 
* Issam Bazzi and James R. Glass. 2000. Modeling outof-vocabulary words for robust speech recognition In Sixth International Conference on Spoken Language Processing, ICSLP 2000 / INTERSPEECH 2000, pages 401–404, Beijing, China 
* Jan A. Botha and Phil Blunsom. 2014. Compositional Morphology for Word Representations and Language Modelling. In Proceedings of the 31st International Conference on Machine Learning (ICML), Beijing, China 
* Rohan Chitnis and John DeNero. 2015. VariableLength Word Encodings for Neural Translation Models. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP) 
* Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning Phrase Representations using RNN Encoder– Decoder for Statistical Machine Translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1724–1734, Doha, Qatar. Association for Computational Linguistics 
* Mathias Creutz and Krista Lagus. 2002. Unsupervised Discovery of Morphemes. In Proceedings of the ACL-02 Workshop on Morphological and Phonological Learning, pages 21–30. Association for Computational Linguistics 
* Nadir Durrani, Hassan Sajjad, Hieu Hoang, and Philipp Koehn. 2014. Integrating an Unsupervised Transliteration Model into Statistical Machine Translation In Proceedings of the 14th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2014, pages 148–153, Gothenburg, Sweden 
* Chris Dyer, Victor Chahuneau, and Noah A. Smith 2013. A Simple, Fast, and Effective Reparameterization of IBM Model 2. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 644–648, Atlanta, Georgia. Association for Computational Linguistics 
* Philip Gage. 1994. A New Algorithm for Data Compression. C Users J., 12(2):23–38, February 
* Barry Haddow, Matthias Huck, Alexandra Birch, Nikolay Bogoychev, and Philipp Koehn. 2015. The Edinburgh/JHU Phrase-based Machine Translation Systems for WMT 2015. In Proceedings of the Tenth Workshop on Statistical Machine Translation, pages 126–133, Lisbon, Portugal. Association for Computational Linguistics 
* Sébastien Jean, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio. 2015. On Using Very Large Target Vocabulary for Neural Machine Translation In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1–10, Beijing, China. Association for Computational Linguistics 
* Nal Kalchbrenner and Phil Blunsom. 2013. Recurrent Continuous Translation Models. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, Seattle. Association for Computational Linguistics 
* Yoon Kim, Yacine Jernite, David Sontag, and Alexander M. Rush. 2015. Character-Aware Neural Language Models. CoRR, abs/1508.06615 
* Philipp Koehn and Kevin Knight. 2003. Empirical Methods for Compound Splitting. In EACL ’03: Proceedings of the Tenth Conference on European Chapter of the Association for Computational Linguistics, pages 187–193, Budapest, Hungary. Association for Computational Linguistics 
* Philipp Koehn, Hieu Hoang, Alexandra Birch, Chris Callison-Burch, Marcello Federico, Nicola Bertoldi, Brooke Cowan, Wade Shen, Christine Moran, Richard Zens, Chris Dyer, Ondˇrej Bojar, Alexandra Constantin, and Evan Herbst. 2007. Moses: Open Source Toolkit for Statistical Machine Translation In Proceedings of the ACL-2007 Demo and Poster Sessions, pages 177–180, Prague, Czech Republic Association for Computational Linguistics 
* Franklin M. Liang. 1983. Word hy-phen-a-tion by com-put-er. Ph.D. thesis, Stanford University, Department of Linguistics, Stanford, CA 
* Wang Ling, Chris Dyer, Alan W. Black, Isabel Trancoso, Ramon Fermandez, Silvio Amir, Luis Marujo, and Tiago Luis. 2015a. Finding Function in Form: Compositional Character Models for Open Vocabulary Word Representation. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1520– 1530, Lisbon, Portugal. Association for Computational Linguistics 
* Wang Ling, Isabel Trancoso, Chris Dyer, and Alan W Black. 2015b. Character-based Neural Machine Translation. ArXiv e-prints, November 
* Thang Luong, Richard Socher, and Christopher D Manning. 2013. Better Word Representations with Recursive Neural Networks for Morphology 
* In Proceedings of the Seventeenth Conference on Computational Natural Language Learning, CoNLL 2013, Sofia, Bulgaria, August 8-9, 2013, pages 104– 113 
* Thang Luong, Hieu Pham, and Christopher D. Manning. 2015a. Effective Approaches to Attentionbased Neural Machine Translation. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1412– 1421, Lisbon, Portugal. Association for Computational Linguistics 
* Thang Luong, Ilya Sutskever, Quoc Le, Oriol Vinyals, and Wojciech Zaremba. 2015b. Addressing the Rare Word Problem in Neural Machine Translation 
* In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 11–19, Beijing, China. Association for Computational Linguistics 
* Tomas Mikolov, Ilya Sutskever, Anoop Deoras, HaiSon Le, Stefan Kombrink, and Jan Cernocký. 2012 Subword Language Modeling with Neural Networks. Unpublished 
* Graham Neubig, Taro Watanabe, Shinsuke Mori, and Tatsuya Kawahara. 2012. Machine Translation without Words through Substring Alignment. In The 50th Annual Meeting of the Association for Computational Linguistics, Proceedings of the Conference, July 8-14, 2012, Jeju Island, Korea - Volume 1: Long Papers, pages 165–174 
* Sonja Nießen and Hermann Ney. 2000. Improving SMT quality with morpho-syntactic analysis. In 18th Int. Conf. on Computational Linguistics, pages 1081–1085 
* Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. 2013. On the difficulty of training recurrent neural networks. In Proceedings of the 30th International Conference on Machine Learning, ICML 2013, pages 1310–1318, Atlanta, USA 
* Maja Popovi´c. 2015. chrF: character n-gram F-score for automatic MT evaluation. In Proceedings of the Tenth Workshop on Statistical Machine Translation, pages 392–395, Lisbon, Portugal. Association for Computational Linguistics 
* Rico Sennrich and Barry Haddow. 2015. A Joint Dependency Model of Morphological and Syntactic Structure for Statistical Machine Translation. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 2081–2087, Lisbon, Portugal. Association for Computational Linguistics 
* Benjamin Snyder and Regina Barzilay. 2008. Unsupervised Multilingual Learning for Morphological Segmentation. In Proceedings of ACL-08: HLT, pages 737–745, Columbus, Ohio. Association for Computational Linguistics 
* David Stallard, Jacob Devlin, Michael Kayser, Yoong Keok Lee, and Regina Barzilay. 2012. Unsupervised Morphology Rivals Supervised Morphology for Arabic MT. In The 50th Annual Meeting of the Association for Computational Linguistics, Proceedings of the Conference, July 8-14, 2012, Jeju Island, Korea - Volume 2: Short Papers, pages 322– 327 
* Miloš Stanojevi´c, Amir Kamran, Philipp Koehn, and Ondˇrej Bojar. 2015. Results of the WMT15 Metrics Shared Task. In Proceedings of the Tenth Workshop on Statistical Machine Translation, pages 256– 273, Lisbon, Portugal. Association for Computational Linguistics Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014 
* Sequence to Sequence Learning with Neural Networks. In Advances in Neural Information Processing Systems 27: Annual Conference on Neural Information Processing Systems 2014, pages 3104–3112, Montreal, Quebec, Canada 
* Jörg Tiedemann. 2009. Character-based PSMT for Closely Related Languages. In Proceedings of 13th Annual Conference of the European Association for Machine Translation (EAMT’09), pages 12–19 
* Jörg Tiedemann. 2012. Character-Based Pivot Translation for Under-Resourced Languages and Domains. In Proceedings of the 13th Conference of the European Chapter of the Association for Computational Linguistics, pages 141–151, Avignon, France 
* Association for Computational Linguistics 
* David Vilar, Jan-Thorsten Peter, and Hermann Ney 2007. Can We Translate Letters? In Second Workshop on Statistical Machine Translation, pages 33–39, Prague, Czech Republic. Association for Computational Linguistics 
* Sami Virpioja, Jaakko J. Väyrynen, Mathias Creutz, and Markus Sadeniemi. 2007. Morphology-Aware Statistical Machine Translation Based on Morphs Induced in an Unsupervised Manner. In Proceedings of the Machine Translation Summit XI, pages 491–498, Copenhagen, Denmark 
* Matthew D. Zeiler. 2012. ADADELTA: An Adaptive Learning Rate Method. CoRR, abs/1212.5701.