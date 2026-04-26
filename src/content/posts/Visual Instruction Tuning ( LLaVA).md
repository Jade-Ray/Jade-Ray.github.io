---
tags:
  - VLM
aliases:
  - LLaVA
title: Visual Instruction Tuning ( LLaVA)
date: 2025-02-01T04:35:26.262Z
draft: false
---
**L**arge **L**anguage **a**nd **V**ision **A**ssistant (LLaVA)，将视觉编码器和 Large Language Model (LLM) 连接，以实现通用的视觉和语言理解。成熟大语言模型在多种下游任务上的适用性及零样本学习和智慧涌现是该方法得以实现的基础。虽然可将视觉语言多模态模型视作 LLM 和对应的视觉指导的简单结合，但一个棘手的问题是缺乏视觉语言多模态指导数据，以及根据视觉输入生成此类数据的范式。
## 多模态视觉语言指令数据
借助类 GPT 文本自回归方法生成视觉语言多模态指令是相当高效且标准，此方法提供了一种将图像文本对话转化为适当指令的格式。
对于图像输入 $X_v$ 和对应的描述 $X_c$，可以自然地创建一组问题 $X_q$，对于描述图像内容的视觉助手来说，一种简单的图像文本对组织格式为

$$
\text{Human: } X_{q} \ X_{v} \text{ <STOP> Assistant: } X_{c} \text{ <STOP>}
$$

上述构建相当简便，但缺乏多样性且在复杂问题上推理不足。
>个人认为此问题可能是由于图像的信息密度远大于文本信息密度导致的，在人类自然语言上表现良好的 LLM 面对图像输入时，一张图像所包含的内容在缺乏文本指示的情况下，可能存在的信息量爆炸（表面信息和深层的推理信息，甚至包括人类根据自身经验独有的偏见信息），强行让 LLM 看图说话要么开始胡言乱语，要么缺乏灵动，机械式的描述图像内容。现有的大模型对高信息密度的图像任无法很好的理解。

该文章采用一种取巧的纯语言图像描述来代替图像输入，将图像内容描述限定在我们希望关注的方向，具体来说，为了实现将图像特征编码到纯文本提示中，使用两种信息表示：
- Captions：描述场景信息。
- Bounding boxes：每个场景中对象类型和空间位置。
![](https://s2.loli.net/2025/01/30/ZeaASVMXWqgNp7Q.png)

## 模型结构及训练方式
使用预训练视觉编码器和 LLM 来高效提高性能，因此模型相当简单。
![](https://s2.loli.net/2025/01/30/nqw1k3cWz8eBJv4.png)
其中，视觉编码器采用预训练的 [CLIP](/posts/contrastive-language-image-pre-trainging/)，LLM 使用 [Vicuna](https://huggingface.co/lmsys/vicuna-7b-v1.5)，对于视觉特征嵌入文本表征空间，使用可学习的权重 $W$ 将视觉特征 $Z_v$ 转换到文本嵌入 $H_v$，形成一系列的视觉 tokens。
尽管使用了视觉输入，但网络本质还是在训练一个 ChatBot $f_{\phi}$，对于每个输入图像 $X_v$，生成 $T$ 轮对话数据 $(X_{q}^{1},X_{a}^{1},\cdots,X_{q}^{T},X_{a}^{T})$，第 $t$ 轮的指令应为：

$$
X_{instruct}^{t} = 
\left\{\begin{matrix} 
  \text{Randomly choose } [X_{q}^{1}, X_{v}] \text{ or } [X_{v}, X_{q}^{1}], \text{ the first turn } t=1 \\  
  X_{q}^{t}, \text{ the remaining turns } t>1 
\end{matrix}\right.
$$

回答 $X_{a}^{t}$ 即为助手 $t$ 轮的回应，下图展示一个两轮的例子：
![](https://s2.loli.net/2025/01/30/Yp4wWJA5kgEnFzr.png)
因此，对于序列长度 $L$ 的目标答案 $X_{a}$，其概率为：

$$
p(X_{a}\mid X_{v}, X_{instruct}) = \prod_{i=1}^{L} p_{\theta}(x_{i}\mid X_{instruct},<i,X_{a},<i)
$$

### 两阶段训练
- 预训练多模态特征对齐。冻结视觉编码器和 LLM，仅训练投影矩阵 $W$，在图像-文本对中实现图像内容注释的最大似然估计。
- 端到端微调。继续冻结视觉编码器，更新投影矩阵 $W$ 和 LLM 预训练权重 $\phi$，通过微调 LLM 来训练多模态 Chatbot。
## 后续提升
### 输入图像分辨率提升
实验表明扩展到高分辨率图像输入，可以提高模型的细节感知能力且有效减小幻觉。一个简单的办法是将图像分为网格输入可以尽可能的保持计算效率。
![](https://s2.loli.net/2025/02/01/rhFM1zbSHiuCyPk.png)
同时，为了保证 LLM 对图像的全局感知，额外添加下采样图像特征。
### 多个任务上的性能提升
![500](https://s2.loli.net/2025/02/01/FM9sackRwYOluGW.png)
实验发现，多模态大模型的综合性能可以概括为模型对各基础任务的组合能力，例如长文本语言推理和较短的视觉问答组合训练可以提高模型多模态写作能力。
![|500](https://s2.loli.net/2025/02/01/RyPUaWLAFTrCKEh.png)
此外，随机下采样训练数据并不会导致模型性能的显著下降，Less is More，数据压缩策略仍有进一步改进的可能。
