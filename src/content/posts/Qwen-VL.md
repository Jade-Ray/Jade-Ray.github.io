---
tags:
  - VLM
aliases:
  - Qwen
title: Qwen-VL
date: 2025-10-06T09:05:25.455Z
draft: false
---
类似于 [KOSMOS-2](/posts/grounding-multimodal-large-language-models-to-the-world/)，Qwen-VL 也通过特殊 tokens 将图像和目标边界框组合到文本中，使文本以指定格式对描述区域图像进行精确理解并生成对应的文本回答，其超文本格式的特殊 tokens 如下图所示：
![|700](https://s2.loli.net/2025/02/05/tizfupVl8UYbOGX.png)
类似的，其中`<img> </img>` 表示图像特征嵌入； `<ref> </ref>` 表示边界框所指的内容； `<box> </box>` 表示对应的目标边界框坐标 $(X_{topleft}, Y_{topleft}),(X_{bottomright}, Y_{bottomright})$，其坐标归一化到 $[0, 1000)$ 内。
## 模型结构
### 图像特征压缩
为了高效减少序列长度，设置一组可学习查询矩阵通过注意力机制关联视觉特征，使用关联后的查询嵌入表示视觉特征可以将特征长度 $(W \times H)^{2}$ 变为自定义的查询长度 $L_{q}$，当然的查询长度太短将导致更多的信息丢失，太长反而会降低收敛速度，论文的消融实验表示对于 $(448, 448)$ 输入图像来说，$L_{q} = 256$ 是一个不错的选择。
### 三阶段管道训练
![image.png](https://s2.loli.net/2025/02/05/C2gVie7tpTswchk.png)
与传统两阶段管道训练方式不同之处在于增加了多任务预训练阶段（阶段 2），该阶段内不使用下采样图像输入（$224 \to 448$），放开所有模块权重并在多个任务（不包含对话等复杂交互式任务）中进行训练，论文表示通过细粒度的语料库和视觉理解可以提高性能。
![image.png](https://s2.loli.net/2025/02/14/BXo38GpQ2OIvLuf.png)
>个人认为性能提升可能来自细粒度语料库带来的基础任务提升，导致的综合任务性能提示，多任务预训练毫无疑问是有效的，但全训练的算力代价有些得不偿失，应该还有更高效的方法。
## Qwen2-VL 性能提升
### 动态分辨率输入
Qwen2-VL 能够处理任意分辨率的图像输入，对于 ViT 输出的视觉 tokens，在相邻的 $2\times2$ 各 tokens 上使用一个简单的 MLP 层压缩为单个 token，因此不同大小图片输入将被动态转换为可变数量的 tokens，最小只占 4 个 tokens。这种设计不仅确保了模型输入与图像原始信息之间的高度一致性，更是模拟了人类视觉感知的自然方式，赋予模型处理任意尺寸图像的强大能力，使其在图像处理领域展现出更加灵活和高效的表现。
![image.png](https://s2.loli.net/2025/02/05/3vEnVXGg7hLuKH4.png)
### Multimodel Rotary Position Embedding (M-ROPE)
通过将原始旋转嵌入分解为代表时间、高度和宽度的三个部分，使得大规模语言模型能够同时捕捉和整合一维文本序列、二维视觉图像以及三维视频的位置信息。
![image.png](https://s2.loli.net/2025/02/11/GD9bTLiVeQP6v5Z.png)
**多模态旋转位置嵌入（M-ROPE）** 的源码注释如下：
![image.png](https://s2.loli.net/2025/02/11/4Gq2na8ER9mPsKe.png)
该方法参考 [Position Embedding](/posts/position-embedding/#多模态-rope) 的思路，取了 $\beta_1=\beta_2=\beta_3=L,\gamma_1=\gamma_2=\gamma_3$ （对于“文本-视频”混合模态），然后视频右段的文本的第一个 Token 的位置，直接取视频最大的位置坐标加 1，牺牲了**对称性**和**等价性**。
## QW2.5-VL 性能提升
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-VL/qwen2.5vl_arc.jpeg)
QW2.5-VL 增强了模型对时间和空间尺度的感知，并进一步简化了网络结构，提高模型效率。
- 细化图像大小感知：在将不同大小图片动态转换为不同长度的 token 后，对于 Visual grouding 不再进行传统的坐标归一化，直接使用图片中的实际尺度尺寸来表示检测框、点等坐标，让模型能直接学习到图片的尺度；
 - 传统 Visual Grounding 方法中使用归一化坐标的主要原因是这避免了输入图像分辨率对模型的影响，同时 $[0,1]$ 坐标轴范围有利于模型收敛和损失计算。但这限制了模型对目标绝对位置的感知，对“远近”、”大小“等绝对尺寸下的概念任务处理困难。且需要额外后处理来做转换。
 - VL 大模型使用视觉动态 token 转换（ViT）后，在 Transformer 中该可变长度视觉 patch token 的序列天然就携带了其在图像中的位置信息（位置编码），并且越是内容复杂、尺寸大的图像所转换的可变 token 就越长；反之就少。这与实际图像中目标大小和绝对位置是契合的，模型只需学习这些 token 与绝对坐标和图像分辨率的对应关系，而不受图像固定分辨率和缩放操作的影响。当然，上述设计需要建立在强大的模型数值分析和回归能力（海量图文数据上训练），并不适用于小模型中（这会带来稳定性问题）。
- 细化视频时间感知：引入动态 FPS 训练和绝对时间编码，将 mRoPE 的 id 与时间快慢对齐，让模型通过时间 id 间隔来学习帧率变化；
- 视觉编码器：从头开始训练了一个原生动态分辨率 ViT，包括 CLIP、视觉语言模型对齐和端到端训练阶段；
- 高效视觉编码器：在 ViT 内使用窗口 Attention 机制，类似 ViTDetect 的方法。
 ![|500](Excalidraw/ViT3D.excalidraw)
 - 窗口注意力数据准备：窗口分割和索引重排，ViT 得到的视觉 Patch 进一步分割为指定尺寸的 Window，通过重排索引快速提取窗口数据（如窗口一的数据为原数据索引 $[0, 1, 5, 6]$），最后将所有重排窗口展平作为新的视觉 Patch （从 $[0, 1, 2, 3, \cdots]$ 到 $[0, 1, 5, 6, \cdots]$），实现不同窗口的并行注意力计算。
 ![|600](Excalidraw/WindowAttenDataArrange.excalidraw)
 - 窗口注意力计算：构建窗口掩码，各窗口只与自己窗口内的数据进行注意力计算，计算复杂度从 $O(N^{2})$ 降为 $O(W^{2} * N/W)$，其中 $W$ 为窗口大小，$N$ 为总序列长度。方法与 Swin Atten 类似，但 Qwen2.5-VL 中将每个窗口视为一种文本序列层面的视觉 token，因此无需各窗口间的交互，在后面多模态 tokens 序列会在 Transformer 框架中进一步关联。
 ![|600](Excalidraw/WindowAttenCal.excalidraw)
