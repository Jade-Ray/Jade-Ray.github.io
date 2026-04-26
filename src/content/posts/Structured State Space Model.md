---
tags:
  - S4
aliases:
  - SSM
  - S4
title: Structured State Space Model
date: 2024-03-14T04:31:10.404Z
draft: false
---
状态空间模型（States Space Model， SSM）是控制理论中一种用状态量描述动态系统的模型，但在深度学习中，探讨的是其中的一个子集，线性时不变系统（系统输出随输入线性变化，且不随时间变化）。该系统对应所有序列信息深度学习任务（视频、音频、文本等），在 RNN 中被广泛使用。

## 线性时不变系统中的 SSM

![](https://s2.loli.net/2024/03/13/qGgbsWnXLFKoVHp.png)

一个简单的连续时不变状态空间模型如上图所示，其中包含三个与时间相关的变量：
- $x(t) \in \mathbb{C}^{n}$ 表示$n$维状态变量；
- $u(t) \in \mathbb{C}^{m}$ 表示$m$维状态输入；
- $y(t) \in \mathbb{C}^{p}$ 表示$p$维输出；
以及四个可学习的矩阵：
- $\mathrm{A} \in \mathbb{C}^{m\times n}$ 是控制状态潜变量$x$的状态矩阵；
- $\mathrm{B} \in \mathbb{C}^{n\times m}$ 是控制矩阵；
- $\mathrm{C} \in \mathbb{C}^{p\times n}$ 是输出矩阵；
- $\mathrm{D} \in \mathbb{C}^{p\times m}$ 是命令矩阵；
整体系统由两个连续微分方程控制：

$$
\begin{align}
x^{\prime}(t) = \mathrm{A}x(t) + \mathrm{B}u(t) \\
y(t) = \mathrm{C}x(t) + \mathrm{D}u(t)
\end{align}
$$

一般而言，系统变量与时间的关系是隐性的（时不变性）$y(t+\tau) = F(x(t+\tau))$，公式可以忽略时间$t$的影响。而且在实践中，$\mathrm{D}u = 0$ 被视为一种轻量化计算的办法。因此，常见的 SSM 连续表示为：

$$
\begin{align}
x^{\prime} &= \mathrm{A}x + \mathrm{B}u \\
y &= \mathrm{C}x
\end{align}
$$

## 连续系统的关键一步———离散化

毫无疑问，如果要求解一个连续系统，我们通常需要将连续输入（或输出）信号离散为“等价的”离散输入（或输出）信号，这是因为离散数据（特征）计算量小，天然带有非线性，鲁棒性强。对于 SSM 而言，整个框架的计算效率由离散化方式决定，这也是大多数 SSM 类框架的区别。这里简单介绍两种离散过程：递归和卷积。

![image.png](https://s2.loli.net/2024/03/13/AxlZrvqRkMVDzOh.png)

### SSM 的递归视图

假设在时间段$[t_{n}, t_{n+1}]$上离散状态变量$x$的差分是其有关时间函数$f$的积分（用梯形面积近似）：

$$
x_{n+1} - x_{n} = \frac{1}{2} \Delta (f(t_{n})+f(t_{n+1})) \text{ , with } \Delta = t_{n+1} - t_{n}
$$

已知状态变量与时间的方程$x^{\prime}_{n} = \mathrm{A}x_{n} + \mathrm{B}u_{n}$，带入$f$ 可得：

$$
\begin{align}
x_{n+1} &= x_{n} + \frac{\Delta}{2} (\mathrm{A}x_{n} + \mathrm{B}u_{n} + \mathrm{A}x_{n+1} + \mathrm{B}u_{n+1}) \\
\Longleftrightarrow x_{n+1} - \frac{\Delta}{2}\mathrm{A}x_{n+1} &= x_{n} + \frac{\Delta}{2}\mathrm{A}x_{n} + \frac{\Delta}{2}\mathrm{B}(u_{n+1} + u_{n}) \\
\overset{u_{n+1} \overset{\Delta}{\simeq} u_{n}}{\Longleftrightarrow} (\mathrm{I} - \frac{\Delta}{2}\mathrm{A})x_{n+1} &= (\mathrm{I} + \frac{\Delta}{2}\mathrm{A})x_{n} + \Delta\mathrm{B}u_{n+1} \\
\Longleftrightarrow x_{n+1} &= (\mathrm{I} - \frac{\Delta}{2}\mathrm{A})^{-1} (\mathrm{I} + \frac{\Delta}{2}\mathrm{A})x_{n} + (\mathrm{I} - \frac{\Delta}{2}\mathrm{A})^{-1} \Delta\mathrm{B}u_{n+1} \\
\end{align}
$$

接下来设置超参：

$$
\begin{align}
\bar{\mathrm{A}} &= (\mathrm{I} - \frac{\Delta}{2}\mathrm{A})^{-1} \left(\mathrm{I} + \frac{\Delta}{2}\mathrm{A}\right) \\
\bar{\mathrm{B}} &= (\mathrm{I} - \frac{\Delta}{2}\mathrm{A})^{-1} \Delta\mathrm{B} \\
\bar{\mathrm{C}} &= \mathrm{C}
\end{align}
$$

可以得到一个清晰的离散化 SSM：

$$
\begin{align}
x_{k} &= \bar{\mathrm{A}} x_{k-1} + \bar{\mathrm{B}} u_{k} \\
y_{k} &= \bar{\mathrm{C}} x_{k}
\end{align}
$$


### SSM 的卷积视图

一般而言，在数字信号处理中卷积操作就是从线性时不变系统中推导的，让我们复习一下。
首先迭代第一个等式：

$$
\begin{align}
\text{Step 0: } x_{0} &= \bar{\mathrm{B}} u_{0} \\
\text{Step 1: } x_{1} &= \bar{\mathrm{A}} x_{0} + \bar{\mathrm{B}} u_{1} = \bar{\mathrm{A}}\bar{\mathrm{B}} u_{0} + \bar{\mathrm{B}} u_{1} \\
\text{Step 2: } x_{2} &= \bar{\mathrm{A}} (\bar{\mathrm{A}}\bar{\mathrm{B}} u_{0} + \bar{\mathrm{B}} u_{1}) + \bar{\mathrm{B}} u_{2} = \bar{\mathrm{A}}^{2}\bar{\mathrm{B}} u_{0} + \bar{\mathrm{A}}\bar{\mathrm{B}} u_{1} + \bar{\mathrm{B}} u_{2} \\
\end{align}
$$

再迭代第二个等式：

$$
\begin{align}
\text{Step 0: } y_{0} &= \bar{\mathrm{C}} x_{0} = \bar{\mathrm{B}} u_{0} \\
\text{Step 1: } y_{1} &= \bar{\mathrm{C}} x_{1} = \bar{\mathrm{C}} (\bar{\mathrm{A}} x_{0} + \bar{\mathrm{B}} u_{1}) = \bar{\mathrm{C}}\bar{\mathrm{A}}\bar{\mathrm{B}} u_{0} + \bar{\mathrm{C}}\bar{\mathrm{B}} u_{1} \\
\text{Step 2: } y_{2} &= \bar{\mathrm{C}} x_{2} = \bar{\mathrm{C}} (\bar{\mathrm{A}}^{2}\bar{\mathrm{B}} u_{0} + \bar{\mathrm{A}}\bar{\mathrm{B}} u_{1} + \bar{\mathrm{B}} u_{2}) = \bar{\mathrm{C}}\bar{\mathrm{A}}^{2}\bar{\mathrm{B}} u_{0} + \bar{\mathrm{C}}\bar{\mathrm{A}}\bar{\mathrm{B}} u_{1} + \bar{\mathrm{C}}\bar{\mathrm{B}} u_{2} \\
\end{align}
$$

通过数学归纳法我们不难看出，$y_{k}$是在$u_{k}$上进行卷积核为$\bar{\mathrm{K}}_{k} = (\bar{\mathrm{C}}\bar{\mathrm{B}},\bar{\mathrm{C}}\bar{\mathrm{A}}\bar{\mathrm{B}},\dots,\bar{\mathrm{C}}\bar{\mathrm{A}}^{k}\bar{\mathrm{B}})$的卷积结果，因此该离散化 SSM：

$$
y = \mathrm{K} \ast u
$$


## SSM 三种视图的优势和局限

![image.png](https://s2.loli.net/2024/03/13/SkuIx4Nz35JdnEs.png)

- 连续视图：
 - 天然适用各种连续数据；
 - 数学分析较为容易；
 - 训练和推理相当漫长；
- 递归视图：
 - 天然的序列数据归纳偏置，并且在上下文中无界；
 - 高效推理（实时状态更新）；
 - 低效训练（无法并行训练）；
 - 对过长数据的训练会梯度爆炸 & 弥散；
- 卷积视图：
 - 可解释的局部特征；
 - 高效训练（可并行化）；
 - 自回归上下文中较为缓慢（需重新计算每个数据点的输入）；
 - 固定的上下文大小；

## 学习矩阵的设置

在上面的分析中，设置矩阵$\bar{\mathrm{C}}$和$\bar{\mathrm{B}}$为可学习的标量是合理的，但矩阵$\bar{\mathrm{A}}$在递归推理中与时间$k$是指数关系，矩阵的指数计算相当耗时。因此，一个固定的矩阵$\bar{\mathrm{A}}$是我们需要的，这同样是大多数 SSM 方法的区别之处。
首先考虑指数计算简单的对角阵：

$$
\mathrm{A} = 
\begin{bmatrix}  \lambda_{1} & 0 & \cdots & 0 \\ 0 & \lambda_{2} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_{n} \end{bmatrix}
\Rightarrow \mathrm{A}^{k} = 
\begin{bmatrix}  \lambda_{1}^{k} & 0 & \cdots & 0 \\ 0 & \lambda_{2}^{k} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_{n}^{k} \end{bmatrix}
$$

从线性代数来看，随机初始化的对角阵$\mathrm{A}$是一个正规矩阵，且高度稀疏的，对预期的泛化结果没有帮助，事实上，与实验结果吻合，该矩阵没有在 SSM 上取得好结果。
[S4论文](https://arxiv.org/abs/2008.07669)中提出一种高阶多项式投影算子，HiPPO 矩阵：

$$
\begin{align}
\mathrm{A} = 
\begin{bmatrix}  1 \\ -1 & 2 \\ 1 & -3 & 3 \\ -1 & 3 & -5 & 4 \\ 1 & -3 & 5 & -7 & 5 \\ -1 & 3 & -5 & 7 & -9 & 6 \\ \vdots &&&&&& \ddots \end{bmatrix} \\
\Rightarrow \mathrm{A}_{nk} = 
\begin{cases}
(-1)^{n-k}(2k+1) & n>k \\
k+1 & n=k \\
0 \\
\end{cases}
\end{align}
$$

该矩阵不是正规矩阵，但是可以分解为一个正规矩阵加一个低秩矩阵，当然论文提供了多种技巧来高效计算 HiPPO，总之，使用 HiPPO 矩阵作为初始化可以在 SSM 上取得相当好的结果。
