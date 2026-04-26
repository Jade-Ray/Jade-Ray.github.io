---
tags:
  - CLIP
aliases:
  - SigLIP
title: Sigmoid Loss for Language Image Pre-Training
date: 2025-02-15T18:07:43.615Z
draft: false
---
回顾 [CLIP#对比学习损失 InfoNCE](/posts/contrastive-language-image-pre-trainging/) 的设计，其特点如下：
- **全局归一化**：通过 Softmax 将正样本对（图像-文本匹配对）的相似度与所有负样本对的相似度进行归一化，迫使正样本在全局对比中脱颖而出。
- **对称计算**：需分别计算图像到文本（Image→Text）和文本到图像（Text→Image）两个方向的损失，导致计算复杂度高且内存占用大。
对于视觉模型 $f(\cdot)$ 和文本模型 $g(\cdot)$，基于 softmax 的对比损失可以表示为：

$$
-\frac{1}{2\mid\mathcal{B}\mid} \sum_{i}^{\mid \mathcal{B} \mid} \left(\overbrace{\log \frac{e^{tx_{i}\cdot y_{i}}}{\sum_{j=1}^{\mid \mathcal{B} \mid} e^{tx_{i}\cdot y_{j}}}}^{\text{image}\to\text{text softmax}} + \overbrace{\log \frac{e^{tx_{i}\cdot y_{i}}}{\sum_{j=1}^{\mid \mathcal{B} \mid} e^{tx_{j}\cdot y_{i}}}}^{\text{text}\to\text{image softmax}} \right)
$$

其中 $x_{i} = \frac{f(I_{i})}{\Vert f(I_{i}) \Vert_{2}}$，$y_{i} = \frac{g(T_{i})}{\Vert g(T_{i}) \Vert_{2}}$，$t$ 是可学习的缩放因子（温度系数）。
## 二分类近似解耦
对比学习的目的是在所有负样本中找到正样本，从匹配过程来看势必要结合所有样本的匹配得分，但若只从匹配结果来看，每个图像-文本匹配任务可以视为独立的二元分类问题，正样本对（匹配）标签为 1，负样本对（不匹配）标签为-1。此时损失函数不再通过全局归一化强制正样本与所有负样本竞争，而是直接最大化正样本对的相似度并最小化负样本对的相似度。其基于 Sigmoid 设计的损失函数为：

$$
-\frac{1}{\mid\mathcal{B}\mid} \sum_{i=1}^{\mid \mathcal{B} \mid} \sum_{j=1}^{\mid \mathcal{B} \mid} \underbrace{\log \frac{1}{1 + e^{z_{ij}(-tx_{i}\cdot y_{j} + b)}}}_{L_{ij}}
$$

其中 $z_{ij} \in \{1, -1\}$ 表示图像-文本是否匹配，$b$ 为可学习偏置项，用来缓解正负样本数量严重不平衡的问题（如负样本数远多于正样本），初始值设为-10 以平衡训练初期的梯度。
## 高效的分块操作
![image.png](https://s2.loli.net/2025/02/16/eRT2EUtSGjqA6cf.png)
- 在分布式训练中，无需再获取全局批次数据以计算 Softmax 分母项，仅需要计算本地 Sigmoid 后交换文本特征即可，通信量降低 90%以上。
- Softmax 需要维护 $\mid \mathcal{B} \mid \times \mid \mathcal{B} \mid$ 的相似性矩阵，而 Sigmoid 仅需要按分块处理本地数据，显存占用从 $O(\mid \mathcal{B} \mid^{2})$ 降至 $O(\mid \mathcal{b} \mid^{2})$，其中 $b$ 为单设备批次大小。
