---
tags:
  - KalmanFilter
aliases: null
title: Kalman filter
date: 2026-01-16T05:37:15.677Z
draft: false
---
在不确定场景中**集合信息 (combining information)** 的通用强大工具。
对于动态系统中不确定信息，我们可以通过概率模型（高斯分布）和统计工具（混合高斯）对该不确定信息进行有根据的猜测。该方法占用内存少，处理速度快。
下面通过对实际问题讨论来理解线性卡尔曼滤波。
## 运动物体位置估计问题
假设运动物体由速度$v$和位置$p$简单建模：

$$
\vec{x} = \begin{bmatrix} 
p\\ 
v 
\end{bmatrix}
$$

### 概率论视角下
现实问题是我们不知道确切的运动物体速度和位置信息，此时可以用高斯分布描述不确定的信息。
![|400](https://s2.loli.net/2023/05/20/p3xsgZWErGFkOCm.png)
如上图所示，均值$\mu$为速度和位置信息的期望值，而 $(\sigma_{p}^{2},\,\sigma_{v}^{2})$ 分别为不确定信息分布的概率。注意此时速度和位置向量是相互独立的（不相关），实际上两者是相关的（基于运动方程），相较于独立分布的向量，我们更关注相关的向量，因为相关性可以提供更多的信息（先验提供边缘分布），减小系统不确定参数数量（降参减小计算量）。
我们可以通过协方差矩阵$\Sigma_{ij}$描述两个变量间线性相关性的强度和尺度，矩阵的每个元素描述第$i$个变量和第$j$个变量之间的相关性（协方差）。
![|400](https://s2.loli.net/2023/05/20/Z37z9DJCEIOpPoQ.png)

现在我们用高斯分布就运动状态（二维向量，包括物体速度与位置）建立概率分布模型，其中$k$表示系统状态时间，均值$\mu = \mathbf{\hat{x}_k}$表示希望得到的最优状态估计，协方差矩阵$\mathbf{P_k}$表示速度概率分布和位置概率分布的相关性。

$$
\begin{equation}
\begin{aligned} 
\mathbf{\hat{x}}_k &= \begin{bmatrix} 
\text{position}\\ 
\text{velocity} 
\end{bmatrix}\\ 
\mathbf{P}_k &= 
\begin{bmatrix} 
\Sigma_{pp} & \Sigma_{pv} \\ 
\Sigma_{vp} & \Sigma_{vv} \\ 
\end{bmatrix} 
\end{aligned} 
\end{equation}
$$

### 估计下一时刻状态
对于物体运动这个动态系统而言，可以通过上一时刻状态来预测下一时刻状态，虽然我们任然不知道真实值在哪，但这不重要，我们将得到新的下一时刻状态概率分布。
![|400](https://s2.loli.net/2023/05/20/WnO6dM8Gb1aPtlr.png)
#### 匀速直线运动
考虑最简单情况，假设物体匀速直线运动，即：

$$
\begin{aligned} 
\color{deeppink}{p_k} &= \color{royalblue}{p_{k-1}} + \Delta t &\color{royalblue}{v_{k-1}} \\ 
\color{deeppink}{v_k} &= &\color{royalblue}{v_{k-1}} 
\end{aligned}
$$

换而言之：

$$
\begin{align} 
\color{deeppink}{\mathbf{\hat{x}}_k} &= \begin{bmatrix} 
1 & \Delta t \\ 
0 & 1 
\end{bmatrix} \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \\ 
&= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}}
\end{align}
$$

其中$\mathbf{F}_k$为预测矩阵，为系统状态到下一时刻的转换矩阵。如下图所示，$\mathbf{F}_k$将改变状态分布，同样，也会改变状态内速度分布和位置分布的相关性（重新投影到预测矩阵上）。
![|400](https://s2.loli.net/2023/05/20/w3zSqvrcL8Jnyoh.png)
对于协方差矩阵，存在推论：

$$
\begin{equation} 
\begin{aligned} 
Cov(x) &= \Sigma\\ 
Cov(\color{firebrick}{\mathbf{A}}x) &= \color{firebrick}{\mathbf{A}} \Sigma \color{firebrick}{\mathbf{A}}^T 
\end{aligned} 
\end{equation}
$$

带入预测矩阵可以得到下一时刻新的状态和协方差矩阵：

$$
\begin{equation} 
\begin{aligned} 
\color{deeppink}{\mathbf{\hat{x}}_k} &= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \\ 
\color{deeppink}{\mathbf{P}_k} &= \mathbf{F_k} \color{royalblue}{\mathbf{P}_{k-1}} \color{white}\mathbf{F}_k^T 
\end{aligned} 
\end{equation}
$$

#### 复杂运动--来自外界的影响
系统变化除状态（位置和速度）自身变化导致外，还可能受到外界影响。例如当运动物体受外力作用时，运动变为更一般的变速直线运动，加速度$\color{darkorange}{a}$将影响下一时刻系统状态：

$$
\begin{aligned} 
\color{deeppink}{p_k} &= \color{royalblue}{p_{k-1}} \color{white} + {\Delta t} &\color{royalblue}{v_{k-1}} + &\frac{1}{2} \color{darkorange}{a} \color{white}{\Delta t}^2 \\ 
\color{deeppink}{v_k} &= &\color{royalblue}{v_{k-1}} + & \color{darkorange}{a} \color{white}{\Delta t} 
\end{aligned}
$$

或矩阵形式：

$$
\begin{equation} 
\begin{aligned} 
\color{deeppink}{\mathbf{\hat{x}}_k} &= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \color{white} + \begin{bmatrix} 
\frac{\Delta t^2}{2} \\ 
\Delta t 
\end{bmatrix} \color{darkorange}{a} \\ 
&= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \color{white} + \mathbf{B}_k \color{darkorange}{\vec{\mathbf{u}_k}} 
\end{aligned} 
\end{equation}
$$

其中$\mathbf{B}_k$称为控制矩阵，$\color{darkorange}{\vec{\mathbf{u}_k}}$称为控制向量（抽象表示为更复杂、更一般的外界影响）。此时，状态更新受外界控制变量加速度$a$影响，一旦加速度确定，该影响相对于时间便是确定的（常量）。
但现实中，复杂的外界控制向量毫无疑问是不确定的，相对于转换矩阵$\mathbf{F}_{k}$可以认为增加了新的不确定噪声，如果将所有噪声可能导致的新分布用概率表示，我们将得到一个高斯分布噪声，均值任然为无噪预测分布的均值，未知的协方差假设为$\color{mediumaquamarine}{\mathbf{Q}_k}$。
![|400](https://s2.loli.net/2023/05/20/vAJQmCPIjoX9keL.png)
该外界不确定噪声将导致预测状态和预测分布协方差改变， 新的预测过程可以写为：
![|400](https://s2.loli.net/2023/05/20/9DxJCPrAa8TE6Yg.png)

$$
\begin{equation} 
\begin{aligned} 
\color{deeppink}{\mathbf{\hat{x}}_k} &= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \color{white} + \mathbf{B}_k \color{darkorange}{\vec{\mathbf{u}_k}} \\ 
\color{deeppink}{\mathbf{P}_k} &= \mathbf{F_k} \color{royalblue}{\mathbf{P}_{k-1}} \color{white}\mathbf{F}_k^T + \color{mediumaquamarine}{\mathbf{Q}_k} 
\end{aligned} 
\end{equation}
$$

简而言之，新的最佳估计是对先前最佳估计的预测，再加上对已知外部影响的修正；新的不确定性是根据旧的不确定性预测，以及一些来自环境的额外不确定性。
### 下一时刻状态观测
预测估计总是伴随者误差，如果不及时通过观测值修正误差，累计误差将导致估计失去意义。一般而言，需要预测的动态系统无法获取准确的状态信息（带噪），假设存在多个传感器可以为我们提供有关系统状态的信息，传感器的衡量指标没有要求（能间接提供状态信息就行）。
首先应该考虑到传感器读数与运动预测单位、比例不同，我们需要通过转换矩阵$\mathbf{H}_{k}$对传感器建模，将运动状态分布投影到传感器读数的分布：
![|600](https://s2.loli.net/2023/05/20/z1NGoY3PkJ5cVeL.png)

$$
\begin{equation} 
\begin{aligned} 
\vec{\mu}_{\text{expected}} &= \mathbf{H}_k \color{deeppink}{\mathbf{\hat{x}}_k} \\ 
\mathbf{\Sigma}_{\text{expected}} &= \mathbf{H}_k \color{deeppink}{\mathbf{P}_k} \color{white} \mathbf{H}_k^T 
\end{aligned} 
\end{equation}
$$

接着我们再看带噪的传感器观测读数，同样用概率分布描述为 $\mathcal{N}(\color{yellowgreen}{\vec{\mathbf{z}_{k}}} ,\color{mediumaquamarine}{\mathbf{R}_k})$，其中方差表示读数的不确定性（噪声），均值为读数的观测值。
![|400](https://s2.loli.net/2023/05/20/BgWalZN59cKhYrQ.png)
### 混合在一起--混合高斯
现在我们得到了两个概率分布，下一时刻预测分布$(\color{fuchsia}{\mu_0}, \color{deeppink}{\Sigma_0}\color{white}) = (\color{fuchsia}{\mathbf{H}_k \mathbf{\hat{x}}_k}, \color{deeppink}{\mathbf{H}_k \mathbf{P}_k \mathbf{H}_k^T}\color{white})$（粉色基于上一时刻）和观测分布$(\color{yellowgreen}{\mu_1}, \color{mediumaquamarine}{\Sigma_1}\color{white}) = (\color{yellowgreen}{\vec{\mathbf{z}_k}}, \color{mediumaquamarine}{\mathbf{R}_k}\color{white})$（绿色基于传感器），希望找到最可能的状态分布，毫无疑问是两个分布的交集，即两个分布的乘积。
![|400](https://s2.loli.net/2023/05/20/XaruMY3xfsUSjLo.png)
#### 混合高斯模型
指多个高斯概率分布的混合分布（乘积），是一个强大且常见的万能近似器，可以看作基于多个先验（潜变量）获取的贝叶斯概率观测值（一个合理的猜测值）。
>存在一个反直觉的推论：两个高斯概率密度函数的乘积产生另一个（未归一化的）高斯概率密度函数。

![|600](https://s2.loli.net/2023/05/20/TdERLGoKWcn3xSv.gif)
推导过程太长不推，新的高斯概率分布是确定的。
![|500](https://s2.loli.net/2023/05/20/pMZreEV4FgqTuHi.png)
对于一维高斯概率分布：

$$
\begin{equation}
\color{orchid}{\mathbf{k}} = \frac{\sigma_0^2}{\sigma_0^2 + \sigma_1^2} 
\end{equation}
$$

$$
\begin{equation} 
\begin{aligned} 
\color{royalblue}{\mu'} &= \mu_0 + \color{orchid}{\mathbf{k}} \color{white}(\mu_1 - \mu_0)\\ 
\color{blueviolet}{\sigma'}^2 &= \sigma_0^2 - \color{orchid}{\mathbf{k}} \color{white}\sigma_0^2 
\end{aligned}
\end{equation}
$$

对于多维高斯概率分布：

$$
\begin{equation}
\color{orchid}{\mathbf{K}} = \Sigma_0 (\Sigma_0 + \Sigma_1)^{-1} 
\end{equation}
$$

$$
\begin{equation} 
\begin{aligned} 
\color{royalblue}{\vec{\mu}'} &= \vec{\mu_0} + \color{orchid}{\mathbf{K}} \color{white}(\vec{\mu_1} - \vec{\mu_0})\\ 
\color{blueviolet}{\Sigma'} &= \Sigma_0 - \color{orchid}{\mathbf{K}} \color{white}\Sigma_0 
\end{aligned} 
\end{equation}
$$

其中$\color{orchid}{\mathbf{K}}$为卡尔曼系数。
#### 下一时刻状态更新
根据高斯混合模型，推导下一时刻状态分布为：

$$
\begin{equation} 
\begin{aligned} 
\mathbf{H}_k \color{royalblue}{\mathbf{\hat{x}}_k'} &= \color{fuchsia}{\mathbf{H}_k \mathbf{\hat{x}}_k} & + & \color{orchid}{\mathbf{K}} ( \color{yellowgreen}{\vec{\mathbf{z}_k}} - \color{fuchsia}{\mathbf{H}_k \mathbf{\hat{x}}_k} ) \\ 
\mathbf{H}_k \color{royalblue}{\mathbf{P}_k'} \color{white}\mathbf{H}_k^T &= \color{deeppink}{\mathbf{H}_k \mathbf{P}_k \mathbf{H}_k^T} & - & \color{orchid}{\mathbf{K}} \color{deeppink}{\mathbf{H}_k \mathbf{P}_k \mathbf{H}_k^T} 
\end{aligned}
\end{equation}
$$

其中卡尔曼系数为：

$$
\begin{equation} 
\color{orchid}{\mathbf{K}} = \color{deeppink}{\mathbf{H}_k \mathbf{P}_k \mathbf{H}_k^T} ( \color{deeppink}{\mathbf{H}_k \mathbf{P}_k \mathbf{H}_k^T} + \color{mediumaquamarine}{\mathbf{R}_k})^{-1} 
\end{equation}
$$

去掉转化矩阵$\mathbf{H}_{k}$返回运动预测单位坐标系中，方便下下时刻状态预测，此时状态分布和系数分别为：

$$
\begin{equation} 
\begin{aligned} 
\color{royalblue}{\mathbf{\hat{x}}_k'} &= \color{fuchsia}{\mathbf{\hat{x}}_k} & + & \color{orchid}{\mathbf{K}'} ( \color{yellowgreen}{\vec{\mathbf{z}_k}} - \color{fuchsia}{\mathbf{H}_k \mathbf{\hat{x}}_k} ) \\ 
\color{royalblue}{\mathbf{P}_k'} &= \color{deeppink}{\mathbf{P}_k} & - & \color{orchid}{\mathbf{K}'} \color{deeppink}{\mathbf{H}_k \mathbf{P}_k} 
\end{aligned}
\end{equation}
$$

$$
\begin{equation} 
\color{orchid}{\mathbf{K}'} = \color{deeppink}{\mathbf{P}_k \mathbf{H}_k^T} ( \color{deeppink}{\mathbf{H}_k \mathbf{P}_k \mathbf{H}_k^T} + \color{mediumaquamarine}{\mathbf{R}_k})^{-1}
\end{equation}
$$

![|500](https://s2.loli.net/2023/05/20/L3pcCzv2HM6bODZ.png)

## 非线性扩展--EKF
