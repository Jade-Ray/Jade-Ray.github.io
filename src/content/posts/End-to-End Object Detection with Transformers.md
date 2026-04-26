---
tags:
  - Transformers
  - DETR
aliases: DETR
title: End-to-End Object Detection with Transformers
date: 2026-01-16T05:36:43.663Z
draft: false
---

## Object Detection
计算机视觉任务中目标检测不同于目标分类（常规分类任务）和目标分割（像素级分类任务），需要从图像特征中预测目标位置，该位置应该是图像的绝对位置，那么绝对位置信息又是从何而来的呢。对于 CNN 而言（平移等变），主流方法解耦绝对位置，变成相对于锚框或者锚点进行局部相对位置的回归，位置信息可以看作是卷积过程中 zero-padding 透露给网络的（使用零填充卷积输入以保持输出尺寸不变，也就是零填充指向了卷积核所在的位置）。对于 Transform 而言（全局注意力），DETR 直接求解位置信息，与 NLP 任务不同的是位置编码是没有顺序的，不需要 masked。（ps.全局注意力矩阵在编码器 self-attn 和解码器 cross-attn 会从平均初始快速稀疏化，最终希望学习到的注意力矩阵必然是高度稀疏的，而注意力模块的动态稀疏将导致网络收敛速度缓慢，此处不过多讨论）
![[DecoderTransforrmVsDETR.excalidraw]]

## Set Prediction
一种目标检测的新范式，不同于之前基于 anchor 的 dense prediction，无需预测大量的 candidates（anchor），直接暴力预测检测框，实现 Sparse 检测器。稀疏预测（取一个大于图像最多预测框个数即可）出目标框看起来运算简单、结构简单，但一个重点需要解决的问题是如何避免预测出几乎相同的目标框（dense prediction 通过海量预测框保证所有可能位置的访问，set prediciton 数量有限的预测框是不能浪费穷举的，要保证它们是置换不变的），DETR 采用匈牙利匹配做二分匹配，获取全局最优的匹配函数，强制每个位置编码学习不同的训练集位置框结构（小框、大框、竖直框、水平框）。（ps. 二分匹配不稳定的缺点也导致了网络收敛速度缓慢，此处不过多讨论）
![[Dense&SetPredictionn.excalidraw]]
每个 set 学习到的位置框结构可视化
![|500](https://s2.loli.net/2023/04/19/b473BwvejmYXJdf.png)

### What do decoder queries essentially mean?
DETR 最初设计采用 set prediciton 解决目标检测问题。因此最初只是并行多个可以学习到位置信息编码的解码对象（每个可学习的 embedding 都没有明确的意义，如上图位置结构可视化所示，各位置结构不同，但不能解释位置的偏好，它是完全随机的。一般为可学习的位置 embeddings 和不可学习初始为零的解码器 embeddings 构成）。但在后续的大多数工作中（deformable-DETR，Conditional-DETR，DAB-DETR 等）将 decoder queries 视作 anchor 位置（二维中心点$(c_x,c_y)$或四维检测框$(c_x,c_y,w,h)$ 类似对应 anchor-free 和 anchor-base）的学习，每一层 decoer 中都会去预测相对偏移量并去更新检测框，得到一个更加精确的检测框并传入下一层。也就是学习下面两个参数：
- good anchors
- relative offset
### What does the cross-attn in decoder actually learn?
Decoder 中的 cross-attention 模块计算图像特征和位置信息编码查询之间的注意力关系，其中，key-value 对来自编码的图像特征和其添加空间位置编码的特征，query 来自位置信息的内容编码。理想状态下，位置信息应该与图像中物体边缘（extremity）区域高度相关，即我们需要一个高质量的内容编码同时匹配 key-value 对中的图像内容编码和空间编码。因此，croos-attn 的每个 head 的 spatial attention map 都在尝试找物体的 extremity 区域。进而实现类似 NMS 的功能。
- 物体内容
- 物体边缘
![|600](https://s2.loli.net/2023/04/22/QDoECIMSR15TV74.png)

## DETR Pipeline
![|700](https://miro.medium.com/v2/resize:fit:720/format:webp/1*RRqclh6R_0yxp8G_yCw5sw.png)
