---
tags:
  - Diffusion
aliases: null
title: Conditioned Generation
date: 2026-01-16T05:36:09.558Z
draft: false
---
生成模型中条件控制生成一般分为两类
- 事后修改（Classifier-Guidance），复用训练好的无条件扩散模型，用一个分类器调整生成过程以实现控制生产，多用于应用层面；
- 事前训练（Classifier-Free），往扩散模型的训练过程中加入条件信号，达到更好的效果，是条件扩散的一般形式，多探索效果上限，缺点是训练成本很高。

## Calssifier-Guidance
对于已经训练好的无条件扩散模型$p(x_{t-1} \mid x_{t})$，一个显著的问题是有条件的扩散过程是否与无条件扩散过程一致。
### 有条件的扩散过程不改变扩散模型
假设有条件的扩散模型$\hat{p}$是马尔科夫过程（条件只影响当前时刻的$x_{t}$，下一时刻由当前时刻加噪而来，与当前时刻条件无关），则：

$$
\begin{aligned}
\hat{p}(x_{0}) &:= p(x_{0}) \\
\hat{p}(y \mid x_{0}) &:= \text{Know labels per sample} \\
\hat{p}(x_{t} \mid x_{t-1}, y) &:= p(x_{t} \mid x_{t-1})  \\
\hat{p}(x_{1:T} \mid x_{0}, y) &:= \prod_{t=1}^{T} \hat{p}(x_{t} \mid x_{t-1}, y)
\end{aligned}
$$

边缘分布不变：

$$
\begin{aligned}
\hat{p}(x_{t} \mid x_{t-1})
&= \int_{y} \hat{p}(x_{t}, y \mid x_{t-1}) \mathrm{d}y \\
&= \int_{y} \hat{p}(x_{t} \mid x_{t-1}, y) \hat{p}(y \mid x_{t-1}) \mathrm{d}y \\
&= \int_{y} \hat{p}(x_{t} \mid x_{t-1}) \hat{p}(y \mid x_{t-1}) \mathrm{d}y \\
&= p(x_{t} \mid x_{t-1}) \int_{y}  \hat{p}(y \mid x_{t-1}) \mathrm{d}y \\
&= p(x_{t} \mid x_{t-1}) \\
\end{aligned}
$$

$$
\begin{aligned}
\hat{p}(x_{t})
&= \int_{x_{0 : t-1}} \hat{p}(x_{0}, \dots, x_{t}) \mathrm{d}x_{0:t-1} \\
&= \int_{x_{0 : t-1}} \hat{p}(x_{0}) \hat{p}(x_{1}, \dots, x_{t} \mid x_{0}) \mathrm{d}x_{0:t-1} \\
&= \int_{x_{0 : t-1}} {p}(x_{0}) {p}(x_{1}, \dots, x_{t} \mid x_{0}) \mathrm{d}x_{0:t-1} \\
&= \int_{x_{0 : t-1}} {p}(x_{0}, \dots, x_{t}) \mathrm{d}x_{0:t-1} \\
&= p(x_{t})
\end{aligned}
$$

根据贝叶斯公式，条件分布也不变：

$$
\begin{aligned}
\hat{p}(x_{t-1} \mid x_{t})
&= \frac{\hat{p}(x_{t-1})\hat{p}(x_{t} \mid x_{t-1})}{\hat{p}(x_{t})} \\
&= \frac{{p}(x_{t-1}){p}(x_{t} \mid x_{t-1})}{{p}(x_{t})} \\
&= p(x_{t-1} \mid x_{t})
\end{aligned}
$$

### 条件输入带来的影响
基于已经训练好的无条件扩散模型$p(x_{t-1} \mid x_{t})$，以$y$作为条件生成可以看作求解$p(x_{t-1} \mid x_{t}, y)$，利用贝叶斯公式可得：

$$
\begin{equation}
p(\boldsymbol{x}_{t-1}|\boldsymbol{y}) = \frac{p(\boldsymbol{x}_{t-1})p(\boldsymbol{y}|\boldsymbol{x}_{t-1})}{p(\boldsymbol{y})}
\end{equation}
$$

补上条件$x_{t}$，得：

$$
\begin{equation}
p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{y}) = \frac{p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)p(\boldsymbol{y}|\boldsymbol{x}_{t-1}, \boldsymbol{x}_t)}{p(\boldsymbol{y}|\boldsymbol{x}_t)}
\end{equation}
$$

注意到扩散前向过程中，$x_{t}$由$x_{t-1}$加噪而来，噪声并不影响分类条件，也就是说$x_{t}$条件下不影响$y$的分布，即$p(\boldsymbol{y}|\boldsymbol{x}_{t-1}, \boldsymbol{x}_t)=p(\boldsymbol{y}|\boldsymbol{x}_{t-1})$，从而：

$$
\begin{equation}
p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{y}) = \frac{p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)p(\boldsymbol{y}|\boldsymbol{x}_{t-1})}{p(\boldsymbol{y}|\boldsymbol{x}_t)} = p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) e^{\log p(\boldsymbol{y}|\boldsymbol{x}_{t-1}) - \log p(\boldsymbol{y}|\boldsymbol{x}_t)}
\end{equation}
$$

注意此时逆扩散后验分布中，条件$y$只与$x_{t-1}$有关，即分母$p(y \mid x_{t})$为常量，分子部分多出一个未知系数$p(y \mid x_{t-1})$。相较于无条件的后验分布$p(x_{t-1}\mid x_{t})$，有条件将引入一个未知系数，该系数为每步扩散结果对条件的后验分布，类似地我们通过网络学习这个分布$p_{\phi}(y\mid x_{t-1})$，这个分类器网络便称为 **Classifer-Guidance**。
### 近似求解系数分布
未知的系数（标签对每步扩散结果的后验分布）无法求解，考虑到扩散模型中步长足够小，使用泰勒展开近似求解，用两种梯度展开的方向。
#### 沿 $x_{t-1} \to x_{t}$ 梯度方向
基于扩散模型所定义的小步长方差表，$x_{t}$应该十分接近$x_{t-1}$，对于这个小范围内概率的变化，我们可以用一阶泰勒展开近似：

$$
\begin{equation}
\log p(\boldsymbol{y}|\boldsymbol{x}_{t-1}) - \log p(\boldsymbol{y}|\boldsymbol{x}_t)\approx (\boldsymbol{x}_{t-1} - \boldsymbol{x}_t)\cdot\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}|\boldsymbol{x}_t)
\end{equation}
$$

接下来，假设原来扩散有$p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)=\mathcal{N}(\boldsymbol{x}_{t-1};\boldsymbol{\mu}(\boldsymbol{x}_t),\sigma_t^2\boldsymbol{I})\propto e^{-\Vert \boldsymbol{x}_{t-1} - \boldsymbol{\mu}(\boldsymbol{x}_t)\Vert^2/2\sigma_t^2}$，那么此时可以得近似解：

$$
\begin{equation}\begin{aligned} 
p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{y}) \propto&\, e^{-\Vert \boldsymbol{x}_{t-1} - \boldsymbol{\mu}(\boldsymbol{x}_t)\Vert^2/2\sigma_t^2 + (\boldsymbol{x}_{t-1} - \boldsymbol{x}_t)\cdot\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}|\boldsymbol{x}_t)} \\ 
\propto&\, e^{-[\Vert \boldsymbol{x}_{t-1} - \boldsymbol{\mu}(\boldsymbol{x}_t)\Vert^2 - 2\sigma_t^2(x_{t-1}-\mu(x_{t})+\mu(x_{t})-x_{t}) \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}|\boldsymbol{x}_t)]/2\sigma_t^2} \\
\\
\propto&\, e^{-\Vert \boldsymbol{x}_{t-1} - \boldsymbol{\mu}(\boldsymbol{x}_t) - \sigma_t^2 \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}|\boldsymbol{x}_t))\Vert^2/2\sigma_t^2} 
\end{aligned}\end{equation}
$$

可以看出 $p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{y})$ 近似于 $\mathcal{N}(\boldsymbol{x}_{t-1};\boldsymbol{\mu}(\boldsymbol{x}_t) + \sigma_t^2 \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}\mid\boldsymbol{x}_t),\sigma_t^2\boldsymbol{I})$，所以新的生成采样即为：

$$
\begin{equation}\boldsymbol{x}_{t-1} = \boldsymbol{\mu}(\boldsymbol{x}_t) \color{skyblue}{+} {\color{skyblue}{\underbrace{\sigma_t^2 \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}\mid\boldsymbol{x}_t)}_{\text{新增项}}}} \color{white}+ \sigma_t\boldsymbol{\varepsilon},\quad \boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})\end{equation}
$$

#### 沿 $x_{t} \to \mu$ 梯度方向
同样考虑到扩散步长的极小量，认为$\log p_{\phi}(y\mid x_{t})$存在相当小的斜率变化，则在$x_{t}=\mu$处泰勒展开近似：

$$
\begin{aligned}
\log p_{\phi}(y \mid x_{t}) 
&\approx \log p_{\phi}(y \mid x_{t})|_{x_{t}=\mu} + (x_{t} -\mu) \cdot\nabla_{x_{t}} \log p_{\phi}(y \mid x_{t})|_{x_{t}=\mu} \\
&= (x_{t} - \mu)g + C_{1}
\end{aligned}
$$

同样，原无条件扩散在高维时有$p(x_{t}\mid x_{t+1})=\mathcal{N}(\mu, \Sigma)$，其高维正态分布似然为

$$
\log p(x_{t}\mid x_{t+1})=-\frac{1}{2} (x_{t} - \mu)^{T} \Sigma^{-1} (x_{t} - \mu) + C
$$

带入待求解后验分布时的近似解为：

$$
\begin{aligned}
\log(p(x_{t}\mid x_{t+1}) p_{\phi}(y\mid x_{t}))
&\approx -\frac{1}{2} (x_{t} - \mu)^{T} \Sigma^{-1} (x_{t} - \mu) + (x_{t} - \mu)g + C_{2} \\
&= -\frac{1}{2} (x_{t} - \mu -\Sigma \, g)^{T} \Sigma^{-1} (x_{t} - \mu - \Sigma \, g) + \frac{1}{2}g^{T}\Sigma \, g + C_{2} \\
&= -\frac{1}{2} (x_{t} - \mu -\Sigma \, g)^{T} \Sigma^{-1} (x_{t} - \mu - \Sigma \, g) + C_{3} \\
&= \log p(z) + C_{4}, \, z \sim \mathcal{N}(\mu + \Sigma \, g, \Sigma)
\end{aligned}
$$

新的生成采样即为：

$$
x_{t-1} = \mu(x_{t}) + {\color{skyblue}{\underbrace{\sigma^{2} \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}\mid\boldsymbol{x}_t)|_{x_{t}=\mu}}_{\text{新增项}}}} \color{white}+ \sigma\boldsymbol{\varepsilon} ,\quad \boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})
$$

>两种近似解的差异为梯度选取的位置$x_{t}$和$\mu(x_{t})$，而一般情况下$\mu(x_{t})$的零阶近似正是$x_{t}$，所以两者的结果是差不多的。

### 不连续情形--DDIM
上面推导出的有条件扩散采样新增项与每步方差$\sigma_{t}$有关，当$\sigma_{t}=0$时（DDIM），新增项为零，带条件的修正失效。此时从 SDE 视角下再看这个问题，根据[SDE-Diffusion](/posts/stochastic-differential-equation-in-diffusion/)中推导的一般前向 SDE：

$$
\begin{equation}d\boldsymbol{x} = \boldsymbol{f}_t(\boldsymbol{x}) dt + g_t d\boldsymbol{w}\end{equation}
$$

对应的一般反向 SDE 为：

$$
\begin{equation}d\boldsymbol{x} = \left(\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}(g_t^2 + \sigma_t^2)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})\right) dt + \sigma_t d\boldsymbol{w}\end{equation}
$$

其中对数似然导数需要改为带条件情况，根据贝叶斯公式得：

$$
\begin{equation}\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}\mid\boldsymbol{y}) = \nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}) + \nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{y}\mid\boldsymbol{x})\end{equation}
$$

使用得分匹配有$\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}) = -\frac{\boldsymbol{\epsilon}{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)}{\bar{\beta}_t}$，因此：

$$
\begin{aligned}
\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}\mid\boldsymbol{y}) &= -\frac{\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)}{\bar{\beta}_t} + \nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{y}\mid\boldsymbol{x})\\ &= -\frac{\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \bar{\beta}_t\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{y}\mid\boldsymbol{x})}{\bar{\beta}_t}
\end{aligned}
$$

![|500](https://s2.loli.net/2023/06/05/VzXcJiNEaRvLWU6.png)

### 超参--梯度缩放因子
原论文（[《Diffusion Models Beat GANs on Image Synthesis》](https://arxiv.org/abs/2105.05233)）发现分类器的梯度缩放因子对生成结果有显著的影响：

$$
\boldsymbol{x}_{t-1} = \boldsymbol{\mu}(\boldsymbol{x}_t) \color{skyblue}{+} \color{skyblue}{\sigma_t^{2} \color{red}{\gamma} \color{skyblue}\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}\mid\boldsymbol{x}_t)} \color{white}+ \sigma_t\boldsymbol{\varepsilon},\quad \boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})
$$

当$\gamma > 1$时，生成过程将使用更多的分类器信号，结果将会提高生成结果与输入条件$y$的相关性，但是会相应地降低生成结果的多样性；反之，则会降低生成结果与输入信号之间的相关性，但增加了多样性。
从原理上讲，可以认为缩放因子是当前条件在所有条件空间中的聚焦程度：

$$
\begin{equation}\tilde{p}(\boldsymbol{y}\mid\boldsymbol{x}_t) = \frac{p^{\gamma}(\boldsymbol{y}\mid\boldsymbol{x}_t)}{Z(\boldsymbol{x}_t)},\quad Z(\boldsymbol{x}_t)=\sum_{\boldsymbol{y}} p^{\gamma}(\boldsymbol{y}\mid\boldsymbol{x}_t)\end{equation}
$$

随着$\gamma$ 的增加，$\hat p(y \mid x_{t})$的预测会越来越接近 one hot 分布，用它来代替$p(y\mid x_t)$作为分类器做 Classifier-Guidance，生成过程会倾向于挑出分类置信度很高的样本。
一种更抽象的理解可以认为不考虑难以解释实际概率意义的$p(y\mid x_{t})$，用生成结果$x_{t-1}$与条件$y$的某个相似或相关度量$sim(x_{t-1}, y)$代替，这个视角下，$\gamma$ 即是控制结果和条件相关性的系数：

$$
p(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{y})\approx \mathcal{N}(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}(\boldsymbol{x}_t) + \sigma_t^2\gamma \nabla_{\boldsymbol{x}_t} \text{sim}(\boldsymbol{x}_t, \boldsymbol{y}),\sigma_t^2\boldsymbol{I})
$$


## Classifier-Free
直接将条件加到生成过程中，定义反向扩散的概率分布为：

$$
p(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_{t}, \boldsymbol{y}) = \mathcal{N}(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}(\boldsymbol{x}_{t}, \boldsymbol{y}),\sigma_{t}^{2}\boldsymbol{I})
$$

类似 DDPM 扩散中对$\mu(x_t)$的参数化表示，$\mu(x_{t},y)$参数化为：

$$
\boldsymbol{\mu}(\boldsymbol{x}_{t}, \boldsymbol{y}) = \frac{1}{\alpha_{t}}\left(\boldsymbol{x}_{t} - \frac{\beta_{t}^{2}}{\bar{\beta}_{t}}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t}, \boldsymbol{y}, t)\right)
$$

训练的损失函数即为：

$$
\mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{y}\sim\tilde{p}(\boldsymbol{x}_0,\boldsymbol{y}), \boldsymbol{\varepsilon}\sim\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})}\left[\left\Vert\boldsymbol{\varepsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\varepsilon}, \boldsymbol{y}, t)\right\Vert^2\right]
$$

它的优点是在训练过程中就引入了额外的输入$y$，理论上输入信息越多越容易训练；它的缺点也是在训练过程中就引入了额外的输入$y$，意味着每做一组信号控制，就要重新训练整个扩散模型。
同样的，classifier-free 方法也可以通过缩放因子$\gamma$控制结果和条件的相关性：

$$
x_{t-1}=\boldsymbol{\mu}(\boldsymbol{x}_t) + \sigma_t^2 \gamma \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}\mid\boldsymbol{x}_t) = \gamma\left[\boldsymbol{\mu}(\boldsymbol{x}_t) + \sigma_t^2 \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}\mid\boldsymbol{x}_t)\right] - (\gamma - 1) \boldsymbol{\mu}(\boldsymbol{x}_t)
$$

通过参数模型直接拟合$\boldsymbol{\mu}(\boldsymbol{x}_t) + \sigma_t^2 \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}\mid\boldsymbol{x}_t)$部分，引入$w=\gamma-1$，新参数模型为条件参数模型和无条件参数模型的加权值：

$$
\begin{aligned}
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, y)
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \sqrt{1 - \bar{\alpha}_t} \; w \nabla_{\mathbf{x}_t} \log p(y \vert \mathbf{x}_t) \\
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) + w \big(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \big) \\
&= (w+1) \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - w \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)
\end{aligned}
$$

换而言之，扩散模型$p_{\theta}(x\mid y)$是基于数据对$(x, y)$训练的，但同时也需要随机丢掉$y$以让扩散模型知道如何生成无条件结果。
