---
tags: []
aliases:
  - Reparameterization
title: Reparameterization Trick
date: 2024-02-22T02:52:20.597Z
draft: false
---
当需要对一些输入变量 $x$ 实现随机变化时，通常使用额外输入 $z$ （类似高斯分布的简单概率分布）来实现。即通过随机采样生成样本 $\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})$，这是不能求参数偏导的（一方面采样是离散的，不可微；另一方面类似高斯分布的概率分布导数没有明确的解析解）。
因此，可以将采样过程重写，即表达随机变量$z$作为确定性变量$\mathbf{z} = \mathcal{T}_\phi(\mathbf{x}, \boldsymbol{\epsilon})$， 其中$\boldsymbol{\epsilon}$是辅助独立随机变量，参数化$\phi$的变换函数$\mathcal{T}_\phi$将随机变量$\boldsymbol{\epsilon}$转换到可对参数$\phi$求导的变量$z$。
假设对随机变量$\boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})$进行简单的线性变换$\mathcal{T}_{\phi}= \boldsymbol{\sigma} \, \boldsymbol{\epsilon}+ \mu$：

$$
\begin{aligned}
\mathbf{z} &\sim q_\phi(\mathbf{z}\vert\mathbf{x}^{(i)}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}^{(i)}, \boldsymbol{\sigma}^{2(i)}\boldsymbol{I}) & \\
\mathbf{z} &= \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} \text{, where } \boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})
\end{aligned}
$$


![Reparameterization | 600](https://lilianweng.github.io/posts/2018-08-12-vae/reparameterization-trick.png)
