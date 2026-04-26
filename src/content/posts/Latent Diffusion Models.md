---
aliases:
  - LDM
tags:
  - Diffusion
title: Latent Diffusion Models
date: 2026-01-16T07:51:32.671Z
draft: false
---
论文 [Simple diffusion: End-to-end diffusion for high resolution images](https://papers.cool/arxiv/2301.11093) 研究表明扩散模型对于高分辨率生成存在固有困难。对于图像生成，无论是像素空间还是潜变量空间，数据分辨率通常都不超过 $64\times64$，过大的分辨率数据如果要获取等同的 FID 等分布指标，必须付出更多的计算代价（训练步数、模型参数、特征维度）。这其实并不符合预期，因为小图通过简单的上采样即可得到 FID 几乎不变的大图（细节缺失），但扩散模型却无法以同等算力得到这个效果。
## Latent 空间扩散优势
LDM 成为扩散模型主流做法的原因可归纳为两个方面。
- 降维的效率更高：假如原始图像的大小为 $512\times512\times3$，直接 patchify 的结果为 $64\times64\times192$，而 LDM 的 Encoder 编码特征为 $64\times64\times4$，降低到了 $1/48$。
- 降低编码特征的方差：LDM 的 Encoder 编码特征具有正则项（[VAE](/posts/variational-auto-encoder/)的 KL 散度或 VQ-VAE 的 VQ 正则化），进一步降低编码特征的方差，避免模型”死记硬背“。
总的来说，”降维+正则“的组合降低了 Latent 的信息量，压缩特征的多样性以提高特征的泛化能力，从而降低了在 Latent 空间学习扩散模型的难度。虽然在特征局部细节上有损，但使用 Perceptual Loss 保证重构数据在宏观上的相似性，使得编码特征的 FID 几乎无损。
## 模型瓶颈——数据信噪比
扩散模型的训练目标是去噪，但 Simple diffusion 观察到，对于加上相同方差的噪声，高分辨率的大图相对小图具有更高的信噪比（SNR）。
![](https://spaces.ac.cn/usr/uploads/2024/04/656907425.png)
图中上面 $512\times512$ 带噪大图通过平均池化下采样到 $64\times64$ 的过程中，比下面直接在小图上添加相同噪声的过程要清晰。说明对于带噪的高分辨率图像其噪声的占比相较小图更低，对于扩散过程反而是更简单的样本，但大图的生成难度显然是更高的，因此导致了模型学习效率的低下。
