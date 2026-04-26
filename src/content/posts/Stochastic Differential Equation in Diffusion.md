---
tags:
  - SDE
aliases: SDE-Diffusion
title: Stochastic Differential Equation in Diffusion
date: 2024-02-01T08:40:36.939Z
draft: false
---
## 随机微分方程（SDE）
首先通过布朗运动介绍 SDE，布朗运动为微粒在液体或气体中做的无规则运动，是一个马尔可夫过程（只与前一个时刻相关），其运动增量可以认为是一个均值为零的正太分布：

$$
B_{t+\Delta t} = B_{t} + \alpha \mathcal{N}(0, \Delta t)
$$

其有如下性质：
- 马尔可夫性：独立增量以及正态增量；
- 处处连续但不可微分：每次的增量来自一个高斯分布的随机采样，它的导数不可定义、没有意义（无法通过古典微积分方法求解）。

对于求解随机微分方程，需要先定义随机过程函数的微元$\Delta B_{t}$（SDE 的解释由构成 SDE 的解提供），下面介绍伊藤积分（Ito）。
假设随机过程为常微分方程加上斜率上的噪音：

$$
\begin{aligned}
\frac{\mathrm{d}X_{t}}{\mathrm{d}t} 
&= f(X_{t}, t) + g(X_{t}, t) * \text{noise} \\
&= f(X_{t}, t) + g(X_{t}, t) * \Delta B_{t} \\
\end{aligned}
$$

”借鉴“积分的方式定义上式形式上的积分：

$$
\begin{aligned}
X_{t} 
&= X_{0} + \int_{0}^{t} f(X_{s}, s) \mathrm{d}s + \color{skyblue} \underbrace{\int_{0}^{t} g(X_{s}, s) \mathrm{d}B_{s}}_{\text{Ito 积分 I}}
\end{aligned}
$$

因为布朗运动的随机过程为独立增量的正太分布，则认为$f(X_{t}, t)=\mu(X_{t}, t)$为漂移系数，$g(X_{t}, t)=\sigma(X_{t}, t)$为扩散系数，该过程称为扩散过程，带入随机微分方程得：

$$
\begin{aligned}
\mathrm{d}X_{t} &= \mu(X_{t}, t) \mathrm{d}t + \sigma(X_{t}, t) \mathrm{d}B_{t} \\
X_{t+s} - X_{t} &= \underbrace{\int_{t}^{t+s} \mu(X_{u}, u) \mathrm{d}u}_{\text{普通勒贝格积分}} + \color{skyblue}\underbrace{\int_{t}^{t+s} \sigma(X_{u}, u) \mathrm{d}B_{u}}_{\text{伊藤积分}} \\
\end{aligned}
$$

其中随机过程的二次变差 $(\mathrm{d}B_{t})^{2}=\sum\limits_{i=0}^{n-1}(B_{t_{i+1}} - B_{t_{i}})^{2}$，其期望$\mathbb{E} \left [\sum\limits_{i=0}^{n-1}(B_{t_{i+1}} - B_{t_{i}})^{2} \right]$ 即高斯噪声方差$\Delta t$的累和$n \Delta t$，因此我们可以形式的记作：

$$
\begin{aligned}
\lim_{n\to\infty}\sum_{i=0}^{n-1}(B_{t_{i+1}}-B_{t_{i} })^{2} &= \lim_{n\to\infty}\sum_{i=0}^{n-1}(t_{i+1}-t_{i)}\\ 
(\mathrm{d}B_{t})^{2} &= \mathrm{d} t
\end{aligned}
$$

解读为布朗运动的二次变差在$[0,T]$上的积累效果与$t$上是一致的。再考虑$\mathrm{d}t \mathrm{d}t, \mathrm{d}t \mathrm{d}B_{t}$ 可知：

| | $\mathrm{d}B_{t}$ | $\mathrm{d}t$ |
| ----------------- | ----------------- | ------------- |
| $\mathrm{d}B_{t}$ | $\mathrm{d}t$ | $0$ |
| $\mathrm{d}t$ | $0$ | $0$ |
则算术布朗运动 Ito 过程中随机微分方程为：

$$
\begin{aligned}
\mathrm{d}X_{t} 
&= \mu(X_{t}, t) \mathrm{d}t + \sigma(X_{t}, t) \mathrm{d}B_{t} \\
&= \mu(X_{t}, t) \mathrm{d}t + \sigma(X_{t}, t) \sqrt{\mathrm{d}t} \\
\end{aligned}
$$

## 扩散过程中的 SDE
基于对[DDPM](/posts/denoising-diffusion-probabilistic-models/)和[DDIM](/posts/denoising-diffusion-implict-models/)中扩散过程的研究，从 SDE 的角度理解，我们可以认为扩散过程为一个在微小时间段内（小的方差表）连续变化的过程，每次的变化是添加一个随机的噪声:

$$
\begin{aligned}
\mathbf{x}_{t+1}
&= \sqrt{1 - \beta_{t}}\mathbf{x}_{t} + \sqrt{ \beta_{t}}\boldsymbol{\epsilon}_{t} \\
\mathbf{x}_{t+1} - \mathbf{x}_{t} = dx
&= (\sqrt{1 - \beta_{t}} - 1)\mathbf{x}_{t}dt + \sqrt{ \beta_{t}}dw
\end{aligned}
$$

此时即可用随机微分方程来描述：

$$
\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t = \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t + g_t \sqrt{\Delta t}\boldsymbol{\varepsilon},\quad \boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})
$$

写成概率形式：

$$
\begin{aligned} 
p(\boldsymbol{x}_{t+\Delta t}\mid\boldsymbol{x}_t) 
=&\, \mathcal{N}\left(\boldsymbol{x}_{t+\Delta t};\boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t, g_t^2\Delta t \,\boldsymbol{I}\right)\\ 
\propto&\, \exp\left(-\frac{\Vert\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t - \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t\Vert^2}{2 g_t^2\Delta t}\right)
\end{aligned}
$$

逆扩散时用贝叶斯公式可得后验：

$$
\begin{aligned} 
p(\boldsymbol{x}_t\mid\boldsymbol{x}_{t+\Delta t}) 
=&\, \frac{p(\boldsymbol{x}_{t+\Delta t}\mid\boldsymbol{x}_t)p(\boldsymbol{x}_t)}{p(\boldsymbol{x}_{t+\Delta t})} = p(\boldsymbol{x}_{t+\Delta t}\mid\boldsymbol{x}_t) \exp\left(\log p(\boldsymbol{x}_t) - \log p(\boldsymbol{x}_{t+\Delta t})\right)\\
\propto&\, \exp\left(-\frac{\Vert\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t - \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t\Vert^2}{2 g_t^2\Delta t} + \log p(\boldsymbol{x}_t) - \log p(\boldsymbol{x}_{t+\Delta t})\right) 
\end{aligned}
$$

对于$\log p(x_{t+\Delta t})$用泰勒展开（两个变量$x_{t}$和$\Delta t$）近似分析$\Delta t$足够小时的情况（此时$p(x_{t+\Delta t} \mid x_{t})$不明显为零）：

$$
\begin{equation}\log p(\boldsymbol{x}_{t+\Delta t})\approx \log p(\boldsymbol{x}_t) + (\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t)\cdot \nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t) + \Delta t \frac{\partial}{\partial t}\log p(\boldsymbol{x}_t)\end{equation}
$$

代入后验公式后，配方可得：

$$
\begin{aligned}
p(\boldsymbol{x}_t\mid\boldsymbol{x}_{t+\Delta t}) 
&\propto \exp\left(-\frac{\Vert\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_{t} - \boldsymbol{f}_{t}(\boldsymbol{x}_{t}) \Delta t\Vert^{2}}{2 g_{t}^{2}\Delta t} - (\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t)\cdot \nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t) + \mathscr{O}(\Delta t)\right) \\
&\propto \exp\left(-\frac{\Vert\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_{t} - \left[\boldsymbol{f}_{t}(\boldsymbol{x}_{t} ) - g_{t}^{2}\nabla_{\boldsymbol{x}_{t}}\log p(\boldsymbol{x}_{t}) \right]\Delta t\Vert^{2}}{2 g_{t}^{2}\Delta t} + \mathscr{O}(\Delta t)\right) \\
\end{aligned}
$$

当$\Delta \to 0$时，概率在时间上的偏微分$\mathscr{O}(\Delta t)$不起作用，因此：

$$
\begin{equation}\begin{aligned} p(\boldsymbol{x}_t\|\boldsymbol{x}_{t+\Delta t}) 
\propto&\, \exp\left(-\frac{\Vert\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t - \left[\boldsymbol{f}_t(\boldsymbol{x}_t) - g_t^2\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t) \right]\Delta t\Vert^2}{2 g_t^2\Delta t}\right) \\ \approx&\,\exp\left(-\frac{\Vert \boldsymbol{x}_t - \boldsymbol{x}_{t+\Delta t} + \left[\boldsymbol{f}_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t}) - g_{t+\Delta t}^2\nabla_{\boldsymbol{x}_{t+\Delta t}}\log p(\boldsymbol{x}_{t+\Delta t}) \right]\Delta t\Vert^2}{2 g_{t+\Delta t}^2\Delta t}\right) 
\end{aligned}\end{equation}|
$$

即后验分布$p(x_{t}\mid x_{t+\Delta t})$近似为一个均值为$\boldsymbol{x}_{t+\Delta t} - \left[\boldsymbol{f}_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t}) - g_{t+\Delta t}^2\nabla_{\boldsymbol{x}_{t+\Delta t}}\log p(\boldsymbol{x}_{t+\Delta t}) \right]\Delta t$、协方差为$g_{t+\Delta t}^2\Delta t\,\boldsymbol{I}$的正态分布，取$\Delta t \to 0$的极限，那么对应的 SDE：

$$
d\boldsymbol{x} = \left[\boldsymbol{f}_t(\boldsymbol{x}) - g_t^2\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}) \right] dt + g_t d\boldsymbol{w}
$$

即逆向扩散过程的随机微分方程。
### 得分匹配
对于确定的逆向 SDE，还需要进一步知道对数似然梯度$\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$来完成生成模型的构建，边缘分布概率模型$p(x_{t})$的$p(x_{t}\mid x_{0})$解析解模型有配分函数：

$$
\begin{equation}p(\boldsymbol{x}_t) = \int p(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\tilde{p}(\boldsymbol{x}_0)d\boldsymbol{x}_0=\mathbb{E}_{\boldsymbol{x}_0}\left[p(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\right]\end{equation}
$$

于是

$$
\begin{equation}\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t) = \frac{\mathbb{E}_{\boldsymbol{x}_0}\left[\nabla_{\boldsymbol{x}_t} p(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\right]}{\mathbb{E}_{\boldsymbol{x}_0}\left[p(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\right]} = \frac{\mathbb{E}_{\boldsymbol{x}_0}\left[p(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\right]}{\mathbb{E}_{\boldsymbol{x}_0}\left[p(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\right]}\end{equation}
$$

其中分母$\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0)$的加权平均形式涉及模型对全体训练样本$x_{0}$的期望，不易计算，因此使用得分匹配方法估计配分函数，通过训练网络模型$s_{\Theta}(x_{t}, t)$学习配分函数，称为得分，其匹配策略为最小化模型对数密度和数据对数密度关于输入的导数之间的平方差期望：

$$
\begin{aligned}
&\,\int \mathbb{E}_{\boldsymbol{x}_0}\left[p(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\left\Vert \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\right\Vert^2\right] d\boldsymbol{x}_t \\ 
=&\, \mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{x}_t \sim p(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\tilde{p}(\boldsymbol{x}_0)}\left[\left\Vert \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\right\Vert^2\right] 
\end{aligned}
$$

回到扩散过程中，我们依然定义扩散过程概率密度为：

$$
\begin{equation} p(\boldsymbol{x}_t|\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_t; \bar{\alpha}_t \boldsymbol{x}_0,\bar{\beta}_t^2 \boldsymbol{I})\end{equation}
$$

根据前面推导扩散过程为 SDE，即求解$p(x_{t+\Delta t}\mid x_{t})$，其存在微分线性解$dx = f_{t}xdt + g_{t}dw$，并且满足：

$$
\begin{equation} p(\boldsymbol{x}_{t+\Delta t}\mid\boldsymbol{x}_0) = \int p(\boldsymbol{x}_{t+\Delta t}\mid\boldsymbol{x}_t) p(\boldsymbol{x}_t\mid\boldsymbol{x}_0) d\boldsymbol{x}_t\end{equation}
$$

我们可以写出

$$
\begin{array}{c|c|c}  
\hline  
\text{记号} & \text{含义} & \text{采样}\\  
\hline  
p(\boldsymbol{x}_{t+\Delta t}\mid\boldsymbol{x}_0) & \mathcal{N}(\boldsymbol{x}_t;\bar{\alpha}_{t+\Delta t} \boldsymbol{x}_0,\bar{\beta}_{t+\Delta t}^2 \boldsymbol{I}) & \boldsymbol{x}_{t+\Delta t} = \bar{\alpha}_{t+\Delta t} \boldsymbol{x}_0 + \bar{\beta}_{t+\Delta t} \boldsymbol{\varepsilon} \\  
\hline  
p(\boldsymbol{x}_t\mid\boldsymbol{x}_0) & \mathcal{N}(\boldsymbol{x}_t;\bar{\alpha}_t \boldsymbol{x}_0,\bar{\beta}_t^2 \boldsymbol{I}) & \boldsymbol{x}_t = \bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\varepsilon}_1 \\  
\hline  
p(\boldsymbol{x}_{t+\Delta t}\mid\boldsymbol{x}_t) & \mathcal{N}(\boldsymbol{x}_{t+\Delta t}; (1 + f_t\Delta t) \boldsymbol{x}_t, g_t^2 \Delta t\, \boldsymbol{I}) & \boldsymbol{x}_{t+\Delta t} = (1 + f_t\Delta t) \boldsymbol{x}_t + g_t\sqrt{\Delta t}\boldsymbol{\varepsilon}_2 \\  
\hline  
{\begin{array}{c}\int p(\boldsymbol{x}_{t+\Delta t}\mid\boldsymbol{x}_t) \\  p(\boldsymbol{x}_t\mid\boldsymbol{x}_0) d\boldsymbol{x}_t\end{array}} &  & {\begin{aligned}&\,\boldsymbol{x}_{t+\Delta t} \\ 
=&\, (1 + f_t\Delta t) \boldsymbol{x}_t + g_t\sqrt{\Delta t} \boldsymbol{\varepsilon}_2 \\  
=&\, (1 + f_t\Delta t) (\bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\varepsilon}_1) + g_t\sqrt{\Delta t} \boldsymbol{\varepsilon}_2 \\  
=&\, (1 + f_t\Delta t) \bar{\alpha}_t \boldsymbol{x}_0 + ((1 + f_t\Delta t)\bar{\beta}_t \boldsymbol{\varepsilon}_1 + g_t\sqrt{\Delta t} \boldsymbol{\varepsilon}_2) \\  \end{aligned}} \\  
\hline  
\end{array}
$$

由此可得：

$$
\begin{equation}\begin{aligned} 
\bar{\alpha}_{t+\Delta t} =&\, (1 + f_t\Delta t) \bar{\alpha}_t \\ 
\bar{\beta}_{t+\Delta t}^2 =&\, (1 + f_t\Delta t)^2\bar{\beta}_t^2 + g_t^2\Delta t \end{aligned}\end{equation}
$$

令$\Delta t \to 0$，分别解得

$$
\begin{equation} 
f_t = \frac{d}{dt} \left(\ln \bar{\alpha}_t\right) = \frac{1}{\bar{\alpha}_t}\frac{d\bar{\alpha}_t}{dt}, \quad g_t^2 = \bar{\alpha}_t^2 \frac{d}{dt}\left(\frac{\bar{\beta}_t^2}{\bar{\alpha}_t^2}\right) = 2\bar{\alpha}_t \bar{\beta}_t \frac{d}{dt}\left(\frac{\bar{\beta}_t}{\bar{\alpha}_t}\right)
\end{equation}
$$

此时对数似然梯度（损失）为：

$$
\begin{aligned}
\nabla_{\boldsymbol{x}_{t}} \log p(\boldsymbol{x}_{t}\mid\boldsymbol{x}_{0}) 
&= \nabla_{\boldsymbol{x}_{t}} \left( -\frac{(\boldsymbol{x}_{t} - \bar{\alpha}_{t}\boldsymbol{x}_{0})^{2}}{2\bar{\beta}_{t}^{2}} + C \right) \\
&= -\frac{\boldsymbol{x}_{t} - \bar{\alpha}_{t}\boldsymbol{x}_{0}}{\bar{\beta}_{t}^{2}} \\
&= -\frac{\bar{\alpha}_{t}\boldsymbol{x}_{0} + \bar{\beta}_{t}\boldsymbol{\varepsilon} - \bar{\alpha}_{t}\boldsymbol{x}_{0}}{\bar{\beta}_{t}^{2}} \\
&= -\frac{\boldsymbol{\varepsilon}}{\bar{\beta}_{t}} \\
\end{aligned}
$$

可学习的得分网络函数定义为$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) = -\frac{\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)}{\bar{\beta}_t}$，最小化匹配损失为：

$$
\begin{equation}
\frac{1}{\bar{\beta}_t^2}\mathbb{E}_{\boldsymbol{x}_0\sim \tilde{p}(\boldsymbol{x}_0),\boldsymbol{\varepsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})}\left[\left\Vert \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\boldsymbol{\varepsilon}, t) - \boldsymbol{\varepsilon}\right\Vert^2\right]
\end{equation}
$$

忽略系数即为 DDPM 的损失函数。
