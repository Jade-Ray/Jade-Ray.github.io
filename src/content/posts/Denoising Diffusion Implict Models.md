---
tags:
  - Diffusion
aliases: DDIM
title: Denoising Diffusion Implict Models
date: 2024-02-01T09:58:44.411Z
draft: false
---
对于 [DDPM](/posts/denoising-diffusion-probabilistic-models/) 而言，它的推导路线可以简单归纳为：

$$
\begin{equation}
p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})\xrightarrow{\text{推导}}p(\boldsymbol{x}_t|\boldsymbol{x}_0)\xrightarrow{\text{推导}}p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)\xrightarrow{\text{近似}}p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)
\end{equation}
$$

可以发现
- 为了求解逆过程概率分布的显式解$p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)$，由于其中$x_{0}$是我们想得到结果，是未知的，我们不得不近似估计一个$x_{0}$，恰好有参数重整化的扩散概率分布$\mathbf{x_{t}} = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon_t}$，通过引入随机量，得到了有意义的解$p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$，但这也被认为是 DDPM 训练速度慢的主要原因之一；
- 逆扩散采样过程只依赖于 $p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$，但采样迭代次数需要与前向扩散时的迭代次数对应以模拟马尔科夫过程，是 DDPM 训练速度慢的主要原因之一；
- 损失函数 $L_t^\text{simple}= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]$ 只依赖于边缘分布 $p(\boldsymbol{x}_t|\boldsymbol{x}_0)$，而不直接依赖于联合分布 $\prod^T_{t=1} p(\mathbf{x}_t \vert \mathbf{x}_{t-1})$。即给定$\bar{\alpha}_t$的值，训练过程就已确定，扩散过程可以不是马尔可夫链。

DDIM 的思想便是既然结果与基于马尔科夫链的前向扩散过程 $p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}, \boldsymbol{x}_0)=p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$ 无关，是否可以在推导过程中换为更一般的分布形式（采样算法）呢？
使用非马尔可夫过程有一个明显的好处是不必拘束于链式法则$p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$的超长迭代次数。

## 非马尔科夫扩散过程
将 DDPM 扩散过程替换为非马尔科夫性质并不改变推导过程。首先我们考虑逆扩散推导中近似逆向马尔科夫分布 $p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)$ 使用的贝叶斯公式：

$$
p(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) 
= p(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ p(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ p(\mathbf{x}_t \vert \mathbf{x}_0) }
$$

其中包括两个边缘分布 $p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)$ 和 $p(\boldsymbol{x}_t|\boldsymbol{x}_0)$，以及前向马尔科夫扩散的概率分布 $p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$。
假设此时为非马尔科夫扩散，不表示 $p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$ 的具体形式，引入超参数$\sigma$设计新的前向扩散过程， 该非马尔科夫过程的联合分布可以表示为：

$$
p_{\sigma}(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = p_{\sigma}(\mathbf{x}_{T} \vert \mathbf{x}_0) \prod^T_{t=2} p_{\sigma}(\mathbf{x}_{t-1} \vert \mathbf{x}_{t}, \mathbf{x}_{0})
$$

该过程$t$时刻的结果任为标准正太分布 $\mathcal{N}$，即不改变边缘分布 $p(\boldsymbol{x}_t|\boldsymbol{x}_0)=\bar{\alpha}_{t}x_{0} + \bar{\beta}_{t}\epsilon$，另一个边缘分布后面可证得与$\sigma$无关，任为 $p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)=\bar{\alpha}_{t-1}x_{0} + \bar{\beta}_{t-1}\epsilon$。
接着再考虑求解上述贝叶斯公式，相较 DDPM，理论上来说非马链过程后验 $p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)$ 的解空间更大，只需要满足边际分布条件： 

$$
\begin{equation}
\int p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) p(\boldsymbol{x}_t|\boldsymbol{x}_0) d\boldsymbol{x}_t = p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)
\end{equation}
$$

通过待定系数法求解这个方 程，考虑 DDPM 已经解出 $p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)$ 为一个正态分布，我们可以更一般的设：

$$
p_{\sigma}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_{t-1}; \kappa_t \boldsymbol{x}_t + \lambda_t \boldsymbol{x}_0, \sigma_t^2 \boldsymbol{I})
$$

 基于上述推导可以列出：

$$
\begin{array}{c|c|c} 
\hline 
\text{记号} & \text{含义} & \text{采样}\\ 
\hline 
p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0) & \mathcal{N}(\boldsymbol{x}_{t-1};\bar{\alpha}_{t-1} \boldsymbol{x}_0,\bar{\beta}_{t-1}^2 \boldsymbol{I}) & \boldsymbol{x}_{t-1} = \bar{\alpha}_{t-1} \boldsymbol{x}_0 + \bar{\beta}_{t-1} \boldsymbol{\varepsilon} \\ 
\hline 
p(\boldsymbol{x}_t|\boldsymbol{x}_0) & \mathcal{N}(\boldsymbol{x}_t;\bar{\alpha}_t \boldsymbol{x}_0,\bar{\beta}_t^2 \boldsymbol{I}) & \boldsymbol{x}_t = \bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\varepsilon}_1 \\ 
\hline 
p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) & \mathcal{N}(\boldsymbol{x}_{t-1}; \kappa_t \boldsymbol{x}_t + \lambda_t \boldsymbol{x}_0, \sigma_t^2 \boldsymbol{I}) & \boldsymbol{x}_{t-1} = \kappa_t \boldsymbol{x}_t + \lambda_t \boldsymbol{x}_0 + \sigma_t \boldsymbol{\varepsilon}_2 \\ 
\hline 
{\begin{array}{c}\int p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) \\ 
p(\boldsymbol{x}_t|\boldsymbol{x}_0) d\boldsymbol{x}_t\end{array}} &  & {\begin{aligned}\boldsymbol{x}_{t-1} =&\, \kappa_t \boldsymbol{x}_t + \lambda_t \boldsymbol{x}_0 + \sigma_t \boldsymbol{\varepsilon}_2 \\ 
=&\, \kappa_t (\bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\varepsilon}_1) + \lambda_t \boldsymbol{x}_0 + \sigma_t \boldsymbol{\varepsilon}_2 \\ 
=&\, (\kappa_t \bar{\alpha}_t + \lambda_t) \boldsymbol{x}_0 + (\kappa_t\bar{\beta}_t \boldsymbol{\varepsilon}_1 + \sigma_t \boldsymbol{\varepsilon}_2) \\ 
\end{aligned}} \\ 
\hline 
\end{array}
$$

其中 $\boldsymbol{\varepsilon},\boldsymbol{\varepsilon}_1,\boldsymbol{\varepsilon}_2\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})$，并根据正太分布叠加性可知 $\kappa_t\bar{\beta}_t \boldsymbol{\varepsilon}_1 + \sigma_t \boldsymbol{\varepsilon}_2\sim \sqrt{\kappa_t^2\bar{\beta}_t^2 + \sigma_t^2} \boldsymbol{\varepsilon}$。对比 $x_{t-1}$ 的两种采样，想要边际分布条件不变，只需满足两个方程：

$$
\begin{equation}\bar{\alpha}_{t-1} = \kappa_t \bar{\alpha}_t + \lambda_t, \qquad\bar{\beta}_{t-1} = \sqrt{\kappa_t^2\bar{\beta}_t^2 + \sigma_t^2}\end{equation}
$$

此时三个未知量，两个方程，拥有较大的解空间，将 $\sigma_t$ 视为可变参数，可得：

$$
\begin{equation}
\kappa_t = \frac{\sqrt{\bar{\beta}_{t-1}^2 - \sigma_t^2}}{\bar{\beta}_t},\qquad \lambda_t = \bar{\alpha}_{t-1} - \frac{\bar{\alpha}_t\sqrt{\bar{\beta}_{t-1}^2 - \sigma_t^2}}{\bar{\beta}_t}
\end{equation}
$$

此时，我们虽然不知道每次前向扩散时的具体分布，但通过待定系数法获得了逆扩散后验分布 $p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)$ 关于自由参数 $\sigma_t$ 的一簇解：

$$
\begin{aligned}
p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_{0}) 
&=\mathcal{N}\left(\boldsymbol{x}_{t-1}; \frac{\sqrt{\bar\beta_{t-1}^{2} - \sigma_{t}^{2}}}{\bar\beta_{t}} \boldsymbol{x}_{t} + \left(\bar\alpha_{t-1} - \frac{\bar\alpha_{t} \sqrt{\bar\beta_{t-1}^{2} - \sigma_{t}^{2}}}{\bar\beta_{t}}\right) \boldsymbol{x}_{0}, \sigma_{t}^{2} \boldsymbol{I}\right) \\
&=\mathcal{N}\left(\boldsymbol{x}_{t-1}; \bar\alpha_{t-1} \boldsymbol{x}_{0} + \frac{\boldsymbol{x}_{t} - \bar\alpha_{t}\boldsymbol{x}_{0}}{\bar\beta_{t}} \sqrt{\bar\beta_{t-1}^{2} - \sigma_{t}^{2}} , \sigma_{t}^{2} \boldsymbol{I}\right) \\
\end{aligned}
$$

>此时证明 $p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)=\bar{\alpha}_{t-1}x_{0} + \bar{\beta}_{t-1}\epsilon$ 边缘分布成立。
>首先回顾高斯分布边缘概率和条件概率关系：
>	假设有关$x$的边缘概率分布为$p(x) = \mathcal{N} (x \vert \mu, \Lambda^{-1})$，其对于$y$的条件概率分布为$p(y \vert x) = \mathcal{N}(y \vert Ax+b, \mathcal{L}^{-1})$，则条件$y$的边缘概率分布为$p(y)=\mathcal{N}(y \vert A \mu + b, \mathcal{L}^{-1} + A\Lambda^{-1}A^{T})$。
>我们已知边缘$x_{t}$的分布为 $p(\boldsymbol{x}_{t}|\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_{t}; \bar{\alpha}_{t} \boldsymbol{x}_{0}, \bar{\beta}_{t} \boldsymbol{I})$，其对于$x_{t-1}$的条件概率分布为$p_{\sigma}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_{t-1}; \kappa_t \boldsymbol{x}_t + \lambda_t \boldsymbol{x}_0, \sigma_t^2 \boldsymbol{I})$，则边缘$x_{t-1}$的分布基于上面关系可得$p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_{t-1}; \kappa_{t} \bar{\alpha}_{t} \boldsymbol{x}_{0} + \lambda_t \boldsymbol{x}_{0}, \, \sigma_t^{2} \boldsymbol{I} + \kappa_{t}^{2} \bar{\beta}_{t} \boldsymbol{I})$。
>带入求得的系数可得均值 $\kappa_{t} \bar{\alpha}_{t} \boldsymbol{x}_{0} + \lambda_t \boldsymbol{x}_{0} = \bar{\alpha}_{t-1} \boldsymbol{x}_{0}$，方差$\sigma_t^{2} \boldsymbol{I} + \kappa_{t}^{2} \bar{\beta}_{t} \boldsymbol{I} = \bar{\beta}_{t} \boldsymbol{I}$。

## 非马尔科夫扩散逆过程采样
同样地，我们希望得到不与 $x_0$ 相关的 $p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$ 采样，如[DDPM](/posts/denoising-diffusion-probabilistic-models/#逆扩散过程)所做，我们通过 $\frac{1}{\bar{\alpha}_t}\left(\boldsymbol{x}_t - \bar{\beta}_t \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right)$ 估计 $x_0$，即重参数化预测噪声$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(x_{t}, t)$方法，其训练所用的目标函数依然是 $\left\Vert\boldsymbol{\varepsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\varepsilon}, t)\right\Vert^2$ （去除权重系数）。将估计的 $x_0$ 带入采样过程的后验公式，得到：

$$
\begin{aligned} 
p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) \approx&\, p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0=\tilde{\boldsymbol{\mu}}(\boldsymbol{x}_t)) \\ 
=&\mathcal{N}\left(\boldsymbol{x}_{t-1}; \frac{\sqrt{\bar\beta_{t-1}^{2} - \sigma_{t}^{2}}}{\bar\beta_{t}} \boldsymbol{x}_{t} + \frac{\bar\alpha_{t-1}}{\bar\alpha_{t}}\left(\boldsymbol{x}_t - \bar{\beta}_t \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right) - \frac{\sqrt{\bar\beta_{t-1}^{2} - \sigma_{t}^{2}}}{\bar\beta_{t}} \left(\boldsymbol{x}_t - \bar{\beta}_t \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right), \sigma_{t}^{2} \boldsymbol{I}\right) \\
=&\mathcal{N}\left(\boldsymbol{x}_{t-1}; \frac{1}{\alpha_{t}}\left(\boldsymbol{x}_t - \bar{\beta}_t \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right) + \sqrt{\bar\beta_{t-1}^{2} - \sigma_{t}^{2}}  \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t), \sigma_{t}^{2} \boldsymbol{I}\right) \\
=&\, \mathcal{N}\left(\boldsymbol{x}_{t-1}; \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \left(\bar{\beta}_t - \alpha_t\sqrt{\bar{\beta}_{t-1}^2 - \sigma_t^2}\right) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right), \sigma_t^2 \boldsymbol{I}\right) \\
or&=\mathcal{N}\left(\boldsymbol{x}_{t-1}; \bar\alpha_{t-1} \boldsymbol{x}_{0} + \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) \sqrt{\bar\beta_{t-1}^{2} - \sigma_{t}^{2}} , \sigma_{t}^{2} \boldsymbol{I}\right) \\
\end{aligned}
$$

其中 $\alpha_t=\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$，相较 DDPM，训练过程（损失）没有变化，但生成（采样）过程包含一个与预测噪声无关的可变动的参数 $\sigma_t$。
### 选取特定 $\sigma_t$ 获得 DDPM 采样
原则上， $\sigma_t$ 可以取任意数，不同的取值会呈现不同的特点，例如当取 $\sigma_{t} = \tilde{\beta}_{t} = \frac{\bar{\beta}_{t-1}\beta_t}{\bar{\beta}_{t}}, \, \beta_t = \sqrt{1 - \alpha_t^2}$ 时有：

$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_t(x_{t})
&= \frac{1}{\alpha_{t}} \left( \boldsymbol{x}_{t} - \left(\bar\beta_{t} - \alpha_{t} \sqrt{\bar{\beta}_{t-1}^{2} - \frac{\bar\beta_{t-1}^{2} \beta_{t}^{2}}{\bar\beta_{t}^{2}}} \right) \boldsymbol{\epsilon}_{\boldsymbol\theta}(\boldsymbol{x}_{t}, t) \right) \\
&= \frac{1}{\alpha_{t}} \left( \boldsymbol{x}_{t} - \left( \frac{\bar\beta_{t}^{2} - \sqrt{\alpha_{t}^{2}\bar\beta_{t-1}^{2}(\bar\beta_{t}^{2} - \beta_{t}^{2})}}{\bar\beta_{t}} \right) \boldsymbol{\epsilon}_{\boldsymbol\theta}(\boldsymbol{x}_{t}, t) \right) \\
&= \frac{1}{\alpha_{t}} \left( \boldsymbol{x}_{t} - \left( \frac{\bar\beta_{t}^{2} - \sqrt{\alpha_{t}^{2} (1 - \bar\alpha_{t-1}^{2})(1 - \bar\alpha_{t}^{2} - 1 + \alpha_{t}^{2})}}{\bar\beta_{t}} \right) \boldsymbol{\epsilon}_{\boldsymbol\theta}(\boldsymbol{x}_{t}, t) \right) \\
&= \frac{1}{\alpha_{t}} \left( \boldsymbol{x}_{t} - \left( \frac{\bar\beta_{t}^{2} - \sqrt{(\alpha_{t}^{2} - \bar\alpha_{t}^{2})(\alpha_{t}^{2} - \bar\alpha_{t}^{2})}}{\bar\beta_{t}} \right) \boldsymbol{\epsilon}_{\boldsymbol\theta}(\boldsymbol{x}_{t}, t) \right) \\
&= \frac{1}{\alpha_{t}} \left( \boldsymbol{x}_{t} - \left( \frac{1 - \bar\alpha_{t}^{2} - 1 + \beta_{t}^{2} + \bar\alpha_{t}^{2}}{\bar\beta_{t}} \right) \boldsymbol{\epsilon}_{\boldsymbol\theta}(\boldsymbol{x}_{t}, t) \right) \\
&= \frac{1}{\alpha_{t}} \left( \boldsymbol{x}_{t} -  \frac{\beta_{t}^{2}}{\bar\beta_{t}} \boldsymbol{\epsilon}_{\boldsymbol\theta}(\boldsymbol{x}_{t}, t) \right) \\
\end{aligned}
$$

$$
p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) \approx p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0=\bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)) = \mathcal{N}\left(\boldsymbol{x}_{t-1}; \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \frac{\beta_t^2}{\bar{\beta}_t}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right),\frac{\bar{\beta}_{t-1}^2\beta_t^2}{\bar{\beta}_t^2} \boldsymbol{I}\right)
$$

即 [DDPM](/posts/denoising-diffusion-probabilistic-models/#193067) 的推导结果。
>通过训练过程没有变化可以看出 DDPM 训练的结果实质上包含了它的任意子序列参数的训练结果。

### 特殊采样 — — Implict Model
上节逆过程采样的后验过程可以写为：

$$
x_{t-1}=\frac{1}{\alpha_t}\left(\boldsymbol{x}_{t} - \left(\bar{\beta}_{t} - \alpha_{t}\sqrt{\bar{\beta}_{t-1}^{2} - \sigma_{t}^{2}}\right) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t}, t)\right) + \sigma_{t}\boldsymbol{\epsilon}_{t}
$$

其中 $\sigma_t$ 有一个相当特殊的取值 $\sigma_{t}= 0$，方程去掉随机项，此时从 $x_t$ 到 $x_{t-1}$ 是一个确定性变换：
$$
\boldsymbol{x}_{t-1} = \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \left(\bar{\beta}_t - \alpha_t \bar{\beta}_{t-1}\right) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right)
$$
也就是说从给定的 $\boldsymbol{x}_T = \boldsymbol{z}$ 出发，基于网络预测的$t$时刻噪声$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$，生成结果 $x_0$ 不带随机性（$\sigma_{t}\boldsymbol{\epsilon}_{t}=0$）。这是一个确定性的隐式概率模型（Implicit），其将任意正态噪声向量基于可训练参数变换为图片的一个确定性变换（类似 GAN）。
#### 从数值方法理解 DDIM--常微分方程（ODE）
上述确定性变换可以等价写为：
$$
\frac{\boldsymbol{x}_t}{\bar{\alpha}_t} - \frac{\boldsymbol{x}_{t-1}}{\bar{\alpha}_{t-1}} = \left(\frac{\bar{\beta}_t}{\bar{\alpha}_t} - \frac{\bar{\beta}_{t-1}}{\bar{\alpha}_{t-1}}\right) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)
$$
当$T$足够大时，或$\alpha_{t}$与$\alpha_{t-1}$足够小时，可以将上式看作为某个常微分方程的差分形式：
$$
\frac{d}{ds}\left(\frac{\boldsymbol{x}(s)}{\bar{\alpha}(s)}\right) = \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{x}(s), t(s)\right)\frac{d}{ds}\left(\frac{\bar{\beta}(s)}{\bar{\alpha}(s)}\right)
$$
假设$s\in[0,1]$，其中$s=0$对应$t=0$，$s=1$对应$t=T$。那么我们现在做的事就是在给定$\boldsymbol{x}(1)\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})$的情况下，去求解出$\boldsymbol{x}(0)$。而 DDPM 或者 DDIM 的迭代过程，对应于该常微分方程的[欧拉方法](https://en.wikipedia.org/wiki/Euler_method)。也就是说，将生成过程等同于求解常微分方程后，可以借助常微分方程的数值解法，为生成过程的加速提供更丰富多样的手段。

>DDIM 的推论到此结束，此时来看还只是提出一种新的逆过程采样模型，没有解决多次迭代导致的训练慢问题。

## 加速采样技巧 — — respacing
关于如何减少$T$步生成采样的时间成本，一个简单的想法是以$[T/S]$间隔采样，将$T$步生成迭代降为$S$步。 此时新的采样序列为$\{\tau_1, \dots, \tau_S\}$，其中$\tau_1 < \tau_2 < \dots <\tau_S \in [1, T] \,and\, S < T$。
则对于训练好的$T$步模型，我们可以当作是以 $\bar{\alpha}_{\tau_1},\bar{\alpha}_{\tau_2},\cdots,\bar{\alpha}_{(\tau_S)}$ 为参数训练的$S$步模型，因此生成采样过程也只需$S$步：
$$
\begin{equation}p(\boldsymbol{x}_{\tau_{i-1}}|\boldsymbol{x}_{\tau_i}) \approx \mathcal{N}\left(\boldsymbol{x}_{\tau_{i-1}}; \frac{\bar{\alpha}_{\tau_{i-1}}}{\bar{\alpha}_{\tau_i}}\left(\boldsymbol{x}_{\tau_i} - \left(\bar{\beta}_{\tau_i} - \frac{\bar{\alpha}_{\tau_i}}{\bar{\alpha}_{\tau_{i-1}}}\sqrt{\bar{\beta}_{\tau_{i-1}}^2 - \tilde{\sigma}_{\tau_i}^2}\right) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_{\tau_i}, \tau_i)\right), \tilde{\sigma}_{\tau_i}^2 \boldsymbol{I}\right)\end{equation}
$$
这就是加速采样的生成过程（$\bar{\alpha}_{\tau}$和$\bar{\beta}_{\tau}$需要重新计算）。为了对比间隔采样与随机性噪声的关系（DDPM 和 DDIM 对噪声分布都很敏感），我们设置 $\sigma_t^2 = \eta \cdot \tilde{\beta}_t$，其中超参$\eta=0$为确定性的 DDIM，$\eta=1$为 DDPM，论文实验结果如下：
![|700](https://s2.loli.net/2023/05/13/OlAtFGZayDgLriz.jpg)

>1. DDPM 训练的结果实质上包含了它的任意子序列参数的训练结果。
>2. 训练过程使用$T$步一方面也许能增强泛化能力，另一方面也允许尝试其他的加速手段。
