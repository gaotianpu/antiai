# 模型的推理部署

* https://github.com/PaddlePaddle/PaddleSlim
* 低比特量化、知识蒸馏、稀疏化和模型结构搜索


LoRA，Low-Rank Adaptation of Large Language Models，大语言模型的低阶适应。具体做法： 冻结预训练好的模型权重参数，然后在每个Transformer（Transforme就是GPT的那个T）块里注入可训练的层，由于不需要对模型的权重参数重新计算梯度，所以，大大减少了需要训练的计算量。

* 2015. [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](./Deep_Compression.md)
* 2017. [A Survey of Model Compression and Acceleration for Deep Neural Networks](./model_compression.md)