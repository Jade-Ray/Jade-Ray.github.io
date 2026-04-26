---
tags:
  - Diffusion
aliases: DDPM
title: Denoising Diffusion Probabilistic Models
date: 2024-02-01T09:03:39.440Z
draft: false
---
定义了一个扩散步骤的马尔科夫链（逐渐将一种分布转化为另一种分布），以缓慢地将随机噪声添加到数据中，然后学习反转扩散过程以从噪声中构建所需的数据样本。与[VAE](/posts/variational-auto-encoder/)或流模型不同，扩散模型是通过固定过程学习的，并且潜在变量具有高维性（与原始数据相同）。
- 前向扩散过程：从目标 $x_{0}$ 添加少量高斯噪声 $T$ 步，熵增扩散，最终 $T \to \infty$，$x_{T}$ 等价于各向独立高斯分布。

$$
p(\boldsymbol{x}_0, \boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_T) = p(\boldsymbol{x}_T|\boldsymbol{x}_{T-1})\cdots p(\boldsymbol{x}_2|\boldsymbol{x}_1) p(\boldsymbol{x}_1|\boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0)
$$

- 逆扩散（降噪）过程：从标准高斯噪声经过 $T$ 时间逆扩散生成真实样本目标 $x_{0}$ 

$$
q(\boldsymbol{x}_0, \boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_T) = q(\boldsymbol{x}_0|\boldsymbol{x}_1)\cdots q(\boldsymbol{x}_{T-2}|\boldsymbol{x}_{T-1}) q(\boldsymbol{x}_{T-1}|\boldsymbol{x}_T) q(\boldsymbol{x}_T)
$$


>理论上来说，无关传统扩散模型的能量模型、得分匹配等概念，也无关 VAE 的变分推断，GAN 的概率散度。大道至简，理解为“渐变模型”更合适。
>简化版的自回归式 VAE（将单步正太分布建模分解为多步正太过程，对于微小变化来说，用正太分布可以足够近似的建模，突破了单步 VAE 的拟合能力限制）。

$$
\begin{equation}
\begin{aligned}
&\text{编码:}\,\,\boldsymbol{x} = \boldsymbol{x}_0 \to \boldsymbol{x}_1 \to \boldsymbol{x}_2 \to \cdots \to \boldsymbol{x}_{T-1} \to \boldsymbol{x}_T = \boldsymbol{z} \\ 
&\text{生成:}\,\,\boldsymbol{z} = \boldsymbol{x}_T \to \boldsymbol{x}_{T-1} \to \boldsymbol{x}_{T-2} \to \cdots \to \boldsymbol{x}_1 \to \boldsymbol{x}_0 = \boldsymbol{x} 
\end{aligned}
\end{equation}
$$


![The Markov chain of forward | 600](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png)

![An Example of training | 600](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/diffusion-example.png)

## 正向扩散过程（概率性的过程）
给定一个从真实数据分布中采样的数据点 $\mathbf{x}_0 \sim q(\mathbf{x})$，它是一个多元高斯随机变量（图像标准化到 $[-1,1]$）。产生一系列噪声样本 $\mathbf{x}_1, \dots, \mathbf{x}_T$，每一步都是随机变量。 步长由方差表 $\{\beta_t \in (0, 1)\}_{t=1}^T$ 控制，随时间变化，类似学习率调整。添加的高斯噪声是随机采样，每次添加会产生不同的扩散效果，讨论它的分布没有意义，因此我们考虑加完噪声的下一时刻数据分布（图像潜分布）。
由于每下一时刻的分布只与当前时刻分布相关，为一个均值 $\sqrt{1 - \beta_t}x_{t-1}$ 方差 $\beta_{t}I$ （考虑到最终 $x_{T}$ 等价为标准正太 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 分布而手设的参数$\beta_t$，新分布的均值与上一时刻数据和当前时刻方差有关，考虑均值与上一时刻有关是因为正太分布的样本均值是无偏量，而样本方差是有偏的 $\frac{-\theta^{2}}{m}$）的各向同性高斯分布：

$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})
$$

其联合概率分布为：

$$
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})=\mathcal{N}(\mathbf{x}_t; \bar{\alpha}_t \mathbf{x}_{0}, \bar{\beta}^2_t\mathbf{I})
$$

>一般而言设计 $\alpha_t^2 + \beta_t^2=1$ 可以减小一半的参数量，并有助于化简形式。考虑到 $\mathcal{N}(\boldsymbol{x}_t;\alpha_t \boldsymbol{x}_{t-1}, \beta_t^2 \boldsymbol{I})$ 意味着 $\boldsymbol{x}_t = \alpha_t \boldsymbol{x}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t,\boldsymbol{\varepsilon}_t\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})$，如果$x_{t-1}$ 也是 $\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})$ 的话，我们希望 $x_t$ 也是 $\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})$，所以确定了 $\alpha_t^2 + \beta_t^2=1$ 

但上述过程的一个问题是计算任意时刻分布 $x_{t}$ 总需要从 $x_{0}$ 迭代计算，不方便。这里使用参数重整化技巧[Reparameterization Trick](/posts/reparameterization-trick/)方便计算任意时刻分布（用标准高斯噪声的随机采样线性表示随机分布，使下一时刻分布变为一个确定量），让$\alpha_t = 1 - \beta_t$和 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$：

$$
\begin{aligned}
\mathbf{x}_t 
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
\end{aligned}
$$

这个公式表明，噪声较大的图像$x_{t}$是噪声较小的图像$x_{t-1}$ 和一些噪声$\epsilon_{t-1}$ 之间的加权平均值，$\beta_{t}$ 的值控制着时间 $t$ 上的噪声量，因此 $\beta_{t}$ 的值被设置的非常低，我们不想让噪声快速主导前向过程。
带入$\mathbf{x}_{t-1} = \sqrt{\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}$：

$$
\begin{aligned}
\mathbf{x}_t 
&= \sqrt{\alpha_{t} \alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{\alpha_{t} - \alpha_{t}\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
\end{aligned}
$$

其中$\boldsymbol{\epsilon}_{t-1}$和$\boldsymbol{\epsilon}_{t-2}$均为标准高斯分布，均值为零，方差为各自前面的乘数项。则此时可以继续使用参数重整化，即用标准高斯分布$\bar{\boldsymbol{\epsilon}}_{t-2}$通过$y = a_{1}\mu_{1} + a_{2}\mu_{2} + \left ( a_{1}^{2}\sigma_{1}^{2} + a_{2}^{2}\sigma_{2}^{2} \right ) \bar{\boldsymbol{\epsilon}}_{t-2}$合并两个高斯分布来转换上式后两项：

$$
\begin{aligned}
\mathbf{x}_t 
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}
\end{aligned}
$$

即任意时刻的先验概率分布（边缘分布）可从目标$x_{0}$直接得到：

$$
q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

通常，当样本噪声越大时，我们可以给予更大的更新步长，所以$\beta_1 < \beta_2 < \dots < \beta_T$ 即 $\bar{\alpha}_1 > \dots > \bar{\alpha}_T$
>相比 VAE（用神经网络学习均值和方差），放弃了模型的编码能力，只得到一个纯粹的生成模型，最终的 $q(\mathbf{x}_t \vert \mathbf{x}_0)$ 可以说与输入 $x_0$ 无关。

### 方差表$\beta_{t}$的选择
再 DDPM 论文中，选取$T=1000,\,\beta_{1}=10^{-4},\,\beta_{T}=0.02$线性插值得到方差表，数值的选取有一定的原因。首先基于设置的超参可以得到调整扩散过程噪声比例的参数$\alpha_{t}$：

$$
\begin{equation}\alpha_t = \sqrt{1 - \frac{0.02t}{T}}\end{equation}
$$

这是一个单调递减的函数，考虑到马尔科夫链中，所有的方法都包括重复、随机地更新直到最后状态开始从均衡分布中采样。而我们无法知道需要运行多少步才能达到均衡分布，只能保证会最终收敛。因此为了保证足够的迭代步数（混合时间），不能在初始就添加大方差的高斯噪声（缩小加噪图和原图的差距，减小欧式距离带来的模糊问题），但也不能一直保持小的步长（当 t 比较大时，接近纯噪声，用欧式距离也无妨），这会导致马尔科夫链在一个峰值附近抽取远超需求的样本（缓慢混合）。
由于希望扩散过程在$T$时刻为纯噪声，因此应该有$\bar{\alpha}_{T} \approx 0$，通过估算：

$$
\begin{equation}\log \bar{\alpha}_T = \sum_{t=1}^T \log\alpha_t = \frac{1}{2} \sum_{t=1}^T \log\left(1 - \frac{0.02t}{T}\right) < \frac{1}{2} \sum_{t=1}^T \left(- \frac{0.02t}{T}\right) = -0.005(T+1)\end{equation}
$$

带入$T=1000$得到的$\bar{\alpha}_T\approx e^{-5}$，符合我们的预期。
#### 余弦方差表
线性方差表的范围$[10^{-4}, 0.02]$与标准化图像像素方差范围$[-1, 1]$关联不高，Improved DDPM 提出基于余弦的方差表：

$$
\beta_t = \text{clip}(1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0.999) \quad\bar{\alpha}_t = \frac{f(t)}{f(0)}\quad\text{where }f(t)=\cos\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big)^2
$$

![image.png|400](https://s2.loli.net/2023/05/18/APkgVlJp4uXId9F.png)
## 逆扩散过程
如果我们可以反转上述过程并从中采样$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$，我们将能够从高斯噪声输入$\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$中重新创建真实样本。 值得注意的是，如果$\beta_t$足够小，可以假设逆扩散过程仍然是一个马尔科夫过程，并且$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$也将是高斯分布的。问题在于我们无法轻易估计$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$，因为它需要使用整个数据集（$x_{t-1}$要向整个数据集的分布采样靠拢）$\{ X_{0} \}$ 作为先验，因此我们需要学习一个模型$p_\theta$近似这些条件概率以运行*反向扩散过程*。
其中高斯分布$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$的均值和方差即为关于$\theta$的可学习变量，因此逆过程的概率分布：

$$
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$

其联合概率分布：

$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \quad
$$

可以得到任意时刻后验的条件（目标$x_{0}$和上一时刻的分布$x_{t}$）概率为：

$$
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{cyan}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{magenta}{\tilde{\beta}_t} \mathbf{I})
$$

使用贝叶斯规则将后验用已知的先验推导：

$$
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) 
= q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ q(\mathbf{x}_t \vert \mathbf{x}_0) }
$$

考虑到三个先验都为正态分布，忽略常数项，只考虑正比于指数部分$\exp(-\frac{1}{2}\frac{(x - \mu)^{2}}{\sigma^{2}})$的项，带入前面得到的任意时刻先验分布公式，其中第一个分布与$x_{0}$无关，均值 $\sqrt{1 - \beta_t}=\sqrt{\alpha_{t}}$ 方差 $\beta_t$，第二个分布为$t-1$时刻先验，均值$\sqrt{\bar \alpha_{t-1}}$方差$1-\bar\alpha_{t-1}$，第三个分布为$t$时刻先验，均值$\sqrt{\bar \alpha_{t}}$方差$1-\bar\alpha_{t}$ :

$$
\begin{aligned}
&\propto \exp \Big(-\frac{1}{2} \big(\frac{(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp \Big(-\frac{1}{2} \big(\frac{\mathbf{x}_t^2 - 2\sqrt{\alpha_t} \mathbf{x}_t \color{cyan}{\mathbf{x}_{t-1}} \color{white}{+ \alpha_t} \color{magenta}{\mathbf{x}_{t-1}^2} }{\beta_t} + \frac{ \color{magenta}{\mathbf{x}_{t-1}^2} \color{white}{- 2 \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0} \color{cyan}{\mathbf{x}_{t-1}} \color{white}{+ \bar{\alpha}_{t-1} \mathbf{x}_0^2}  }{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp\Big( -\frac{1}{2} \big( \color{magenta}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 - \color{cyan}{(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \color{white}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)}
\end{aligned}
$$

常数项$C$不包含当前分布$x_{t-1}$可以省略，剩下部分为关于当前分布$x_{t-1}$的一元二次变换，继续通过*参数重整化*用高斯分布$\mathcal{N}(\tilde{\boldsymbol{\mu}}_{t}, \tilde{\beta}_t)$表示确定性变量$x_{t-1}$。一元二次方程存在定理$ax^{2} + bx = a(x + \frac{b}{2a})^{2} + c$，则重整化后的高斯分布方差为$\frac{1}{a}$（因为指数项内放大分布的精度，即方差的倒数）均值为$\frac{b}{2a}$ （回想$\alpha_t = 1 - \beta_t$和$\bar{\alpha}_t = \prod_{i=1}^T \alpha_i$）：

$$
\begin{aligned}
\tilde{\beta}_t 
&= 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) 
= 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})})
= \color{lime}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
\tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0)
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \\
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0) \color{lime}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0\\
\end{aligned}
$$

我们得到了 $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$ 的显式解，但不是我们想要的答案，因为我们只想通过 $x_t$ 来预测 $x_{t-1}$，而不依赖 $x_0$。
>如果我们能够通过 $x_t$ 来预测 $x_0$，那么就可以消去上式显式解的 $x_0$，只依赖于 $x_t$，这部分可以看作是训练一个去噪模型 $\Vert \boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)\Vert^2$，从带噪数据预测原始数据。

具体来说，由已知的先验的参数重整化转换$\mathbf{x_{t}} = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon_t},\boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ 可推得 $\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t)$ 带入上式：

$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_t(x_{t})
&= \frac{\sqrt{\alpha_{t}}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}} \mathbf{x}_{t} + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_{t}} \frac{1}{\sqrt{\bar{\alpha}_{t}}}(\mathbf{x}_{t} - \sqrt{1 - \bar{\alpha}_{t}}\boldsymbol{\epsilon}_{t}) \\
&= {\frac{1}{\sqrt{\alpha_{t}}} \Big( \frac{\alpha_{t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_{t}} \mathbf{x}_{t} + \frac{\beta_{t}}{1-\bar\alpha_{t}} \mathbf{x}_{t} - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t(x_{t}, t) \Big)} \\
&= {\frac{1}{\sqrt{\alpha_{t}}} \Big( \frac{1 - \beta_{t} -\bar\alpha_{t} + \beta_{t}}{1-\bar\alpha_{t}} \mathbf{x}_{t} - \frac{1 - \alpha_{t}}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t(x_{t}, t) \Big)} \\
&= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t(x_{t}, t) \Big)}
\end{aligned}
$$

最终，我们通过估计 $x_0$ 近似推断了生成过程： ^193067

$$
\begin{equation} 
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) \approx q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0=\bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)) = \mathcal{N}\left(\boldsymbol{x}_{t-1}; \frac{1}{\sqrt\alpha_t}\left(\boldsymbol{x}_t - \frac{\beta_t}{\sqrt{\bar{\beta}_t}}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right),\frac{\bar{\beta}_{t-1}\beta_t}{\bar{\beta}_t} \boldsymbol{I}\right)
\end{equation}
$$

>对于 $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$，建模成均值向量可学习的正太分布 $\mathcal{N}(\boldsymbol{x}_{t-1};\boldsymbol{\mu}(\boldsymbol{x}_t), \sigma_t^2 \boldsymbol{I})$。此处我们预估了 $x_t$ 预测 $x_0$ 的结果来使 $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$ 近似 $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$，当然开始的预估是相当不准的，每次的 $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ 便是一小步的修正，以此逐步获得更为精细的解。

### 方差 $\boldsymbol{\Sigma}_\theta$ 
现在我们已经知道了如何近似求解逆扩散过程中后验分布的均值，该讨论这个近似后验分布的方差是什么：
- 不可学习的常量：基于后验分布的公式可以得到与$x_{0}$无关的方差 $\tilde{\beta}_t=\color{lime}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t}$，该方差略小于先验方差 $\beta_{t}$，但再 DDPM 论文中使用先验 $\beta_{t}$ 近似 $\tilde{\beta}_t$ （两个方差的选取皆可表明后验分布方差具有一定的鲁棒性）；
- 可学习变量：通过网络学习获得方差 $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \exp(\mathbf{v} \log \beta_t + (1-\mathbf{v}) \log \tilde{\beta}_t)$，由 Improved DDPM 论文提出，损失计算改为$L_\text{hybrid} = L_\text{simple} + \lambda L_\text{VLB}$，其中引入$\lambda L_\text{VLB}$与可学习方差$\boldsymbol{\Sigma}_\theta$产生可调节的关联。
## 最大似然损失
考虑到逆向生成$x_0$的模型$p_{\theta}(x)$满足最大似然准则，参考[Variational Auto-Encoder](/posts/variational-auto-encoder/#70ddde)的近似变分下界公式，取负数获得负对数似然的上界：

$$
\begin{aligned}
- \log p_\theta(\mathbf{x}_0) 
&\leq - \log p_\theta(\mathbf{x}_0) + D_\text{KL}(q(\mathbf{x}_{1:T}\vert\mathbf{x}_0) \| p_\theta(\mathbf{x}_{1:T}\vert\mathbf{x}_0) ) \\
&= -\log p_\theta(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T}) / p_\theta(\mathbf{x}_0)} \Big] \\
&= -\log p_\theta(\mathbf{x}_0) + \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} + \log p_\theta(\mathbf{x}_0) \Big] \\
&= \mathbb{E}_q \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big]
\end{aligned}
$$

对左右两边同时乘以目标$x_0$的期望$\mathbb{E}_{q(\mathbf{x}_0)}$，左边变为交叉熵$H(q(x_{0}), p_{\theta})$，右边期望从$x_{1:T}$变为$x_{0:T}$可得交叉熵的上界：

$$
\text{Let }L_\text{VLB} 
= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \geq - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0)
$$

进一步化简交叉熵损失，重写为几个 KL 散度和熵的组合：

$$
\begin{aligned}
L_\text{VLB} 
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&= \mathbb{E}_q \Big[ \log\frac{\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{ p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t) } \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=1}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \Big( \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\cdot \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)} \Big) + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{q(\mathbf{x}_1 \vert \mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]\\
&= \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \Big] \\
&= \mathbb{E}_q [\underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} ]
\end{aligned}
$$

其中第二行是联合概率密度的展开，第四行将$t1$时刻的散度提到外面，第五行将$q(x_{t} \mid x_{t-1})$ 用贝叶斯规则展开，第七行对$\log$求和进行裂项相消。最后一行中，$L_T$是常数，在训练期间可以忽略，因为扩散过程$q$没有可学习的参数，并且$\mathbf{x}_T$是高斯噪声。$L_{0}$即为$L_{t-1}$取$t=1$时的结果，两者可以合并（$L_{0}$可以看作是连续空间到离散空间的解码 loss）。
对$L_{t-1}$项的散度计算为扩散过程 $q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$ 符合高斯分布对逆扩散过程 $p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$ 符合高斯分布。

>对于两个单一变量的高斯分布$p$和$q$而言，它们的$KL$散度为 $KL(p,q) = \log \frac{\sigma_{2}}{\sigma_{1}} + \frac{\sigma_{1}^{2} + (\mu_{1} - \mu_{2})^{2}}{2\sigma_{2}^{2}} - \frac{1}{2}$ 

又因为将逆扩散过程 $p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ 的方差也设置成与$\beta$相关的常数，则散度计算的第一项和第三项常数可以省略。因此我们可以对后验分布均值$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)$ 进行建模，构建含参$\theta$的网络来预测后验分布的期望值。进而 $L_{t-1} \,or\, L_{t}$ 项的$KL$散度可以化简为：

$$
L_{t-1} = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2 \| \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) \|^2_2} \| \color{cyan}{\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)} - \color{magenta}{\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)}\color{white} \|^{2} \Big]+C
$$

此时我们可以设计一个$D_{\theta}$网络来学习这个散度分布（含未知量$x_0$），其输入是$x_t$和时间编码$t$，输出取决于我们的建模目标：
1. 输出等于前向过程的后验分布均值$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)$，即预测后验分布的期望；
2. 输出等于未知量$x_0$，即直接预测原始数据（任然需要马尔科夫多次迭代获得最终高质量生成样本）；
3. 基于扩散过程中已知的边缘分布关系$\mathbf{x_{t}} = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon_t},\boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$可以用$x_t$近似估计未知量$x_0$，将未知量变为噪声$\epsilon$，网络输出等于$\epsilon$，即预测随机变量（噪声）。
论文采用第三种方法。也就是说，在逆扩散过程 $p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$ 中，我们的目标是训练 $\boldsymbol{\mu}_\theta(\mathbf{x}_{t} (\mathbf{x_{0}}, \boldsymbol{\epsilon}), t)$ 来尽可能接近扩散过程中的 $\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)$，其中$x_{t}$为训练时的输入不含参，但${\epsilon}_t$ 为$t$时刻的随机高斯噪声采样无法用来训练（对未知的随机采样$\epsilon$求参数$\theta$的偏导是违反直觉的），故重新参数重整化高斯噪声使随机变量$\epsilon$ 转换为确定性变量${\epsilon}_\theta(\mathbf{x}_t, t)$ 来预测随机噪声：

$$
\begin{aligned}
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) &= \color{lime}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big)} \\
\text{Thus }\mathbf{x}_{t-1} &= \mathcal{N}(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
\end{aligned}
$$

则最小化与$\tilde{\boldsymbol{\mu}}$差异的损失项$L_{t}$为：

$$
\begin{aligned}
L_t 
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2  \|\boldsymbol{\Sigma}_\theta \|^2_2} \| \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)} - \color{magenta}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \Big)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big] 
\end{aligned}
$$

论文作者发现，去掉常数项权重可以使扩散模型表现更好，因此简化后的损失即为：

$$
\begin{aligned}
L_t^\text{simple}
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
\end{aligned}
$$

总损失即可写为：

$$
L_\text{simple} = L_t^\text{simple} + C
$$


## 实例化
根据简化后的最大似然损失，可以设计下面的伪代码：

![DDPM Algor | 700](https://s2.loli.net/2023/03/26/He9lXWPruUJNxtE.png)
可以看到，采样过程需要采样$T$次来模拟马尔科夫过程（逆向马尔科夫扩散），使网络可以学习到每次采样过程$t$时的噪声分布。同样的，需要相当大的采样次数（使马尔科夫均衡分布）将导致网络训练时间漫长。
