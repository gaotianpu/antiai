"""
bpe is short for Byte Pair Encoder. It translates arbitrary utf-8 strings into
sequences of integers, where each integer represents small chunks of commonly
occuring characters. This implementation is based on openai's gpt2 encoder.py:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
but was mildly modified because the original implementation is a bit confusing.
I also tried to add as many comments as possible, my own understanding of what's
going on.

bpe 是字节对编码器的缩写。 它将任意 utf-8 字符串转换为整数序列，其中每个整数代表一小块经常出现的字符。 
这个实现是基于openai的gpt2 encoder.py: https://github.com/openai/gpt-2/blob/master/src/encoder.py
但是稍微修改了一下，因为原来的实现有点混乱。我也试过 添加尽可能多的评论，我自己对发生的事情的理解。

openAI的rust实现：https://github.com/openai/tiktoken

代码注释：
https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py
"""

import os
import json
import regex as re
import requests

import torch

# -----------------------------------------------------------------------------

def bytes_to_unicode():
    """
    Every possible byte (really an integer 0..255) gets mapped by OpenAI to a unicode character that represents it visually. 
    Some bytes have their appearance preserved because they don't cause any trouble. 
    These are defined in list bs. For example:
    chr(33) returns "!", so in the returned dictionary we simply have d[33] -> "!".
    However, chr(0), for example, is '\x00', which looks ugly. So OpenAI maps these bytes, into new characters in a range where chr() returns a single nice character.
    So in the final dictionary we have d[0] -> 'Ā' instead, which is just chr(0 + 2**8).
    In particular, the space character is 32, which we can see by ord(' '). 
    Instead, this function will shift space (32) by 256 to 288, so d[32] -> 'Ġ'.
    So this is just a simple one-to-one mapping of bytes 0..255 into unicode characters that "look nice", either in their original form, or a funny shifted character like 'Ā', or 'Ġ', etc.

    每个可能的字节（实际上是一个整数 2的8次方, 0..255）都被 OpenAI 映射到一个 unicode 字符，以直观地表示它。
    有些字节保留了它们的外观，因为它们不会造成任何麻烦。 这些在列表 bs 中定义。 例如： chr(33) 返回“!”，所以在返回的字典中我们只有 d[33] -> “!”。
    但是，例如chr(0)是'\x00'，看起来很难看。 所以 OpenAI 将这些字节映射到一个范围内的新字符，其中 chr() 返回一个很好的字符。
    所以在最终的字典中，我们有 d[0] -> 'Ā'，它只是 chr(0 + 2**8)。
    特别是空格字符是32，我们可以通过ord(' ')看到。
    相反，此函数会将空间 (32) 移动 256 到 288，因此 d[32] -> 'Ġ'。
    所以这只是一个简单的一对一映射，将字节 0..255 映射到“看起来不错”的 unicode 字符，无论是原始形式，还是有趣的移位字符，如“Ā”或“Ġ”等。
    """
    # the 188 integers that render fine in their original form and need no shifting
    # 188 个整数以其原始形式呈现良好并且不需要移位
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:] #?为啥要来这么一下,复制一份？
    # print(sorted(bs))
    # print(sorted(cs))
    # all integers b in bs will simply map to chr(b) in the output dict
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # bs 中的所有整数 b 将简单地映射到输出字典中的 chr(b) 
    # 现在获取其他 68 个需要移位的整数的表示 
    # 每个都将得到映射的 chr(256 + n)，其中 n 将在循环中从 0...67 增长
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            # if this byte is "ugly" then map it to the next available "nice" character
            # 如果这个字节是“丑陋的”，则将其映射到下一个可用的“漂亮”字符
            cs.append(2**8+n)
            # print(n, 2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    # ord 字符->ASCII数值, chr ASCII数值->字符
    # print(cs)
    d = dict(zip(bs, cs)) #key:idx, value:char(nice char)

    # tmp = {k: v for k, v in sorted(d.items(), key=lambda item: item[0])}
    # print(tmp,ord('a'),ord('b'),ord("A"))

    print(ord("中") , [v for v in "中".encode('utf-8')] ) #,tuple(ord("中")) #中文需要转ascii码？

    return d

def get_pairs(word):
    """
    Return all bigrams as a set of tuples, of consecutive elements in the iterable word.
    将所有二元组作为可迭代单词中连续元素的一组元组返回。
    """
    pairs = set() #不能重复？
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:

    def __init__(self, encoder, bpe_merges):
        # byte encoder/decoder
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        # bpe token encoder/decoder
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        # bpe merge list that defines the bpe "tree", of tuples (a,b) that are to merge to token ab
        # bpe 合并列表，定义要合并到标记 ab 的元组 (a,b) 的 bpe“树”
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # the splitting pattern used for pre-tokenization
        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions <-- original openai comment
        # 用于预标记化的拆分模式 应该添加 re.IGNORECASE 以便 BPE 合并可以发生在大写版本的收缩 <-- 原始 openai 评论
        """
        ok so what is this regex looking for, exactly?
        python re reference: https://docs.python.org/3/library/re.html
        - the vertical bars | is OR, so re.findall will chunkate text as the pieces match, from left to right
        - '\'s' would split up things like Andrej's -> (Andrej, 's)
        - ' ?\p{L}': optional space followed by 1+ unicode code points in the category "letter"
        - ' ?\p{N}': optional space followed by 1+ unicode code points in the category "number"
        - ' ?[^\s\p{L}\p{N}]+': optional space, then 1+ things that are NOT a whitespace, letter or number
        - '\s+(?!\S)': 1+ whitespace characters (e.g. space or tab or etc) UNLESS they are followed by non-whitespace
                       so this will consume whitespace characters in a sequence but exclude the last whitespace in
                       that sequence. that last whitespace has the opportunity to then match the optional ' ?' in
                       earlier patterns.
        - '\s+': 1+ whitespace characters, intended probably to catch a full trailing sequence of whitespaces at end of string
        So TLDR:
        - we are special casing a few common apostrophe constructs ('s, 't, 're, ...) and making those into separate tokens
        - we then separate out strings into consecutive chunks of 1) letters, 2) numbers, 3) non-letter-numbers, 4) whitespaces

        好吧，这个正则表达式到底在寻找什么？
         python重新参考：https://docs.python.org/3/library/re.html
         - 竖线 | 是 OR，因此 re.findall 将在片段匹配时从左到右对文本进行分块
         - '\'s' 会拆分 Andrej's -> (Andrej, 's)
         - ' ?\p{L}'：可选空格后跟类别“字母”中的 1+ 个 unicode 代码点
         - ' ?\p{N}'：可选空格后跟类别“数字”中的 1+ unicode 代码点
         - ' ?[^\s\p{L}\p{N}]+'：可选空格，然后是 1+ 个不是空格、字母或数字的东西
         - '\s+(?!\S)'：1+ 个空白字符（例如空格或制表符等），除非它们后跟非空白字符
                        所以这将消耗序列中的空白字符但排除最后一个空白字符
                        那个序列。 最后一个空格有机会匹配可选的“？” 在
                        较早的模式。
         - '\s+'：1+ 个空白字符，可能用于捕获字符串末尾的完整尾随空白序列
         所以 TLDR：
         - 我们对一些常见的撇号结构（'s、't、're、...）进行特殊封装，并将它们制成单独的标记
         - 然后我们将字符串分成连续的块，包括 1) 字母、2) 数字、3) 非字母数字、4) 空格

        python官方re库会出现下面的错误，需要用到第三方reg库： import regex as re
        re.error: bad escape \p at position 26
        
        """
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def bpe(self, token):
        """
        this function uses self.bpe_ranks to iteratively merge all the possible bpe tokens up the tree. 
        token is a string of one individual 'word' (after regex tokenization) and after byte encoding, e.g. 'Ġthere'.
        此函数使用 self.bpe_ranks 迭代地将所有可能的 bpe 标记合并到树中。 
        令牌是一个单独的“单词”（在正则表达式标记化之后）和字节编码之后的字符串，例如 'Ġthere'。
        """
        # token is a string of one individual 'word', after byte encoding, e.g. 'Ġthere'
        # 令牌是一个单独的“单词”的字符串，经过字节编码，例如 'Ġthere'

        # memoization, for efficiency 记忆，为了效率, 缓存粒度？
        if token in self.cache:
            return self.cache[token]

        word = tuple(token) # individual characters that make up the token, in a tuple 组成令牌的单个字符，在一个元组中
        pairs = get_pairs(word) # get all bigrams 得到所有二元组
        print("pairs",pairs)

        if not pairs:
            return token

        while True:

            # find the next lowest rank bigram that can be merged
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break # no more bigrams are eligible to be merged
            first, second = bigram

            # we will now replace all occurences of (first, second) in the list of current
            # words into one merged token first_second, in the output list new_words
            # 我们现在将当前单词列表中所有出现的 (first, second) 替换为输出列表 new_words 中的一个合并标记 first_second
            new_word = []
            i = 0
            while i < len(word):
                # find the next occurence of first in the sequence of current words
                # 在当前单词的序列中找到下一次出现的 first
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                # if this occurence is also followed by second, then merge them into one
                # 如果此事件之后也有第二个，则将它们合并为一个
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # all occurences of (first, second) have been merged to first_second
            # 所有出现的 (first, second) 都已合并到 first second
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        # concat all words into a string, and use ' ' as the separator. 
        # Note that by now all characters have been byte encoded, 
        # guaranteeing that ' ' is not used in the actual data and is a 'special' delimiter character
        # 将所有单词连接成一个字符串，并使用 ' ' 作为分隔符。 请注意，到目前为止，所有字符都已进行字节编码，
        # 保证 ' ' 未在实际数据中使用，并且是一个“特殊”分隔符
        word = ' '.join(word)

        # cache the result and return
        self.cache[token] = word
        return word

    def encode(self, text):
        """ string goes in, list of integers comes out
        字符串进入，整数列表出来 """
        bpe_idx = []
        # pre-tokenize the input text into string tokens (words, roughly speaking)
        # 将输入文本预标记为字符串标记（单词，粗略地说）
        tokens = re.findall(self.pat, text)
        
        # process each token into BPE integers 将每个标记处理成 BPE 整数
        for token in tokens:
            # encode the token as a bytes (b'') object  将令牌编码为字节 (b'') 对象
            token_bytes = token.encode('utf-8') #
            # translate all bytes to their unicode string representation and flatten
            # 将所有字节转换为它们的 unicode 字符串表示并展平
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            
            # perform all the applicable bpe merges according to self.bpe_ranks
            # 根据 self.bpe_ranks 执行所有适用的 bpe 合并
            token_merged = self.bpe(token_translated).split(' ')
            # translate all bpe tokens to integers 将所有 bpe 标记转换为整数
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            # extend our running list of all output integers 扩展我们所有输出整数的运行列表
            bpe_idx.extend(token_ix)
        return bpe_idx

    def encode_and_show_work(self, text):
        """ debugging function, same as encode but returns all intermediate work 
        调试功能，与编码相同，但返回所有中间工作 """
        bpe_idx = []
        parts = []
        tokens = re.findall(self.pat, text)
        print("tokens:",tokens)
        for token in tokens:
            token_bytes = token.encode('utf-8')
            # print("token_bytes:",token_bytes)
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            
            token_merged = self.bpe(token_translated).split(' ')
            print(token,token_translated,token_merged)
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
            parts.append({
                'token': token,
                'token_bytes': token_bytes,
                'token_translated': token_translated,
                'token_merged': token_merged,
                'token_ix': token_ix,
            })
        out = {
            'bpe_idx': bpe_idx, # the actual output sequence
            'tokens': tokens, # result of pre-tokenization
            'parts': parts, # intermediates for each token part
        }
        return out

    def decode(self, bpe_idx):
        """ list of integers comes in, string comes out
        整数列表进来，字符串出来 """
        # inverse map the integers to get the tokens 反向映射整数以获取标记
        tokens_merged = [self.decoder[token] for token in bpe_idx]
        # inverse the byte encoder, e.g. recovering 'Ġ' -> ' ', and get the bytes 
        # 反转字节编码器，例如 恢复 'Ġ' -> ' '，并获取字节
        tokens_flat = ''.join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        # recover the full utf-8 string 恢复完整的 utf-8 字符串
        text = tokens_bytes.decode('utf-8', errors='replace')
        return text

def get_file(local_file, remote_file):
    """ downloads remote_file to local_file if necessary """
    if not os.path.isfile(local_file):
        print(f"downloading {remote_file} to {local_file}")
        response = requests.get(remote_file)
        open(local_file, "wb").write(response.content)

def get_encoder():
    """
    Returns an instance of the GPT BPE Encoder/Decoder and handles caching of "database" files.
    """
    home_dir = os.path.expanduser('~')
    cache_dir = os.path.join(home_dir, '.cache', 'mingpt')
    os.makedirs(cache_dir, exist_ok=True)

    # load encoder.json that has the raw mappings from token -> bpe index
    encoder_local_file = os.path.join(cache_dir, 'encoder.json')
    encoder_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
    get_file(encoder_local_file, encoder_remote_file)
    with open(encoder_local_file, 'r') as f:
        encoder = json.load(f)
    # 256 individual byte tokens, 50,000 merged tokens, and 1 special <|endoftext|> token
    assert len(encoder) == 50257 

    # load vocab.bpe that contains the bpe merges, i.e. the bpe tree structure in the form tuples (a, b), that indicate that (a, b) is to be merged to one token ab
    # 加载包含 bpe 合并的 vocab.bpe，即元组 (a, b) 形式的 bpe 树结构，表示 (a, b) 将合并为一个标记 ab
    vocab_local_file = os.path.join(cache_dir, 'vocab.bpe')
    vocab_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
    get_file(vocab_local_file, vocab_remote_file)
    with open(vocab_local_file, 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    # light postprocessing: strip the version on first line and the last line is a blank
    # 轻度后处理：去掉第一行的版本，最后一行是空白
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    assert len(bpe_merges) == 50000 # 50,000 merged tokens

    # construct the Encoder object and return
    enc = Encoder(encoder, bpe_merges)
    return enc

# -----------------------------------------------------------------------------

class BPETokenizer:
    """ PyTorch-aware class that wraps the Encoder above 
    包装上面编码器的 PyTorch 感知类 """

    def __init__(self):
        self.encoder = get_encoder()

    def __call__(self, text, return_tensors='pt'):
        # PyTorch only; here because we want to match huggingface/transformers interface
        assert return_tensors == 'pt'
        # single string input for now, in the future potentially a list of strings
        assert isinstance(text, str)
        # encode and create a "batch dimension" of 1
        idx = [self.encoder.encode(text)]
        # wrap into PyTorch tensor
        out = torch.tensor(idx, dtype=torch.long)
        return out

    def decode(self, idx):
        # ensure a simple 1D tensor for now
        assert idx.ndim == 1
        # decode indices to text
        text = self.encoder.decode(idx.tolist())
        return text


# bpe paper codes begin
import collections

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

def process():
    vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,
    'n e w e s t </w>':6, 'w i d e s t </w>':3}

    num_merges = 12
    for i in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(best)
        print(vocab)
        print('~~~')

# bpe paper codes end

if __name__ == '__main__':
    # process()  ##bpe paper codes
    # bytes_to_unicode()
    # print(get_pairs("hell,world"))

    # here is an encoding example
    text = "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗 中国加油！"
    e = get_encoder()
    r = e.encode_and_show_work(text)
    # print(r)

    # print("Original text is:")
    # print(text)
    # print("First the text gets pre-tokenized, broken up into chunks, the outcome is:")
    # print(r['tokens'])
    # # ['Hello', '!!', ' I', "'m", ' Andrej', ' Karpathy', '.', ' It', "'s", ' 2022', '.', ' w', '00', 't', ' :', 'D', ' 🤗']
    # print("Then we iterate over each chunk and process them in turn...")
    # for part in r['parts']:
    #     print(part)
    # # {'token': 'Hello', 'token_bytes': b'Hello', 'token_translated': 'Hello', 'token_merged': ['Hello'], 'token_ix': [15496]}
    # # {'token': '!!', 'token_bytes': b'!!', 'token_translated': '!!', 'token_merged': ['!!'], 'token_ix': [3228]}
    # # {'token': ' I', 'token_bytes': b' I', 'token_translated': 'ĠI', 'token_merged': ['ĠI'], 'token_ix': [314]}
    # # {'token': "'m", 'token_bytes': b"'m", 'token_translated': "'m", 'token_merged': ["'m"], 'token_ix': [1101]}
    # # {'token': ' Andrej', 'token_bytes': b' Andrej', 'token_translated': 'ĠAndrej', 'token_merged': ['ĠAndre', 'j'], 'token_ix': [10948, 73]}
    # # {'token': ' Karpathy', 'token_bytes': b' Karpathy', 'token_translated': 'ĠKarpathy', 'token_merged': ['ĠK', 'arp', 'athy'], 'token_ix': [509, 5117, 10036]}
    # # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # # {'token': ' It', 'token_bytes': b' It', 'token_translated': 'ĠIt', 'token_merged': ['ĠIt'], 'token_ix': [632]}
    # # {'token': "'s", 'token_bytes': b"'s", 'token_translated': "'s", 'token_merged': ["'s"], 'token_ix': [338]}
    # # {'token': ' 2022', 'token_bytes': b' 2022', 'token_translated': 'Ġ2022', 'token_merged': ['Ġ2022'], 'token_ix': [33160]}
    # # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # # {'token': ' w', 'token_bytes': b' w', 'token_translated': 'Ġw', 'token_merged': ['Ġw'], 'token_ix': [266]}
    # # {'token': '00', 'token_bytes': b'00', 'token_translated': '00', 'token_merged': ['00'], 'token_ix': [405]}
    # # {'token': 't', 'token_bytes': b't', 'token_translated': 't', 'token_merged': ['t'], 'token_ix': [83]}
    # # {'token': ' :', 'token_bytes': b' :', 'token_translated': 'Ġ:', 'token_merged': ['Ġ:'], 'token_ix': [1058]}
    # # {'token': 'D', 'token_bytes': b'D', 'token_translated': 'D', 'token_merged': ['D'], 'token_ix': [35]}
    # # {'token': ' 🤗', 'token_bytes': b' \xf0\x9f\xa4\x97', 'token_translated': 'ĠðŁ¤Ĺ', 'token_merged': ['ĠðŁ', '¤', 'Ĺ'], 'token_ix': [12520, 97, 245]}
    # # (refer to the code inside Encoder.encode for what these intermediates are)
    # print("and the final outcome is concatenating and flattening all the token_ix:")
    # print(r['bpe_idx'])
    # # [15496, 3228, 314, 1101, 10948, 73, 509, 5117, 10036, 13, 632, 338, 33160, 13, 266, 405, 83, 1058, 35, 12520, 97, 245]
    # # this would then become the integer input sequence to the transformer
    # print("ready to feed into a Transformer!")
