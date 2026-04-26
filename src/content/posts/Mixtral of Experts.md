---
tags: []
aliases:
  - MoE
title: Mixtral of Experts
date: 2025-02-15T09:51:39.013Z
draft: false
---
Transformer 模型由 Attention 层和 MLP 层组成，MoE 替换模型中的 MLP 层，进而希望减少 MLP 层的计算量。考虑到一般的 MLP 层 FFN (FeedForward Network) 定义：

$$
\boldsymbol{y} = f(\boldsymbol{x}\boldsymbol{W}^{(A)})\boldsymbol{W}^{(B)}
$$

其中 $x\in \mathbb{R}^{d}$ 是输入向量，$\boldsymbol{W}^{(A)}\in \mathbb{R}^{d\times D},\boldsymbol{W}^{(B)}\in \mathbb{R}^{D\times d}$ 是两个参数矩阵，$f$ 是 Element-wise 的激活函数。假设 $n$ 是一个能整除 $D$ 的整数，那么上述可以等价地用分块矩阵写成：

$$
\boldsymbol{y} = f\big(\boldsymbol{x}\begin{bmatrix}\boldsymbol{W}^{(A)}_1 & \boldsymbol{W}^{(A)}_2 & \cdots & \boldsymbol{W}^{(A)}_n\end{bmatrix}\big)\begin{bmatrix}\boldsymbol{W}^{(B)}_1 \\ \boldsymbol{W}^{(B)}_2 \\ \vdots \\ \boldsymbol{W}^{(B)}_n\end{bmatrix} = \sum_{i=1}^n \underbrace{f(\boldsymbol{x}\boldsymbol{W}^{(A)}_i)\boldsymbol{W}^{(B)}_i}_{\boldsymbol{v}_i}
$$

其中 $\boldsymbol{W}^{(A)}_i = \boldsymbol{W}^{(A)}_{[:,(i-1)c:ic]}, \boldsymbol{W}^{(B)}_i = \boldsymbol{W}^{(B)}_{[(i-1)c:ic,:]},c= D/n$，简单来说 FFN 可以等价表示成 $n$ 个向量 $v_{1},v_{2},\cdots,v_{n}$ 之和，每个向量代表了一个小模型 $f(\boldsymbol{x}\boldsymbol{W}^{(A)}_i)\boldsymbol{W}^{(B)}_i$ 的输出，每个小模型计算量相当，这些小模型就是 MoE 中的 “Expert”。
>MoE 设计的思路便是：能否只挑 $k$ 个向量的和来逼近 $n$ 个向量的和呢？这样就可以将计算量降低到 $k/n$ 了。

## Sparse 模型
直接给出 MoE 的一般形式：

$$
\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{p}} p_i \boldsymbol{v}_i
$$

即训练一个对各 Expert 排序打分的模型（即 Router），取 $top_{k}$ 的 $p_{i}$ 求和后作为 $k=n$ 时 Dense 模型的近似。
### 一种几何视角解释
求解近似解可以看作是最小化近似解与解的差值：

$$
\mathop{\text{argmin}}_{\lambda_1,\lambda_2,\cdots,\lambda_n\in\{0,1\}}\left\Vert\sum_{i=1}^n \lambda_i \boldsymbol{v}_i - \sum_{i=1}^n\boldsymbol{v}_i\right\Vert^2\quad\text{s.t.}\quad \sum_{i=1}^n \lambda_i = k
$$

记 $\gamma_i = 1 - \lambda_i$，得：

$$
\mathop{\text{argmin}}_{\gamma_1,\gamma_2,\cdots,\gamma_n\in\{0,1\}}\left\Vert\sum_{i=1}^n \gamma_i \boldsymbol{v}_i\right\Vert^2\quad\text{s.t.}\quad \sum_{i=1}^n \gamma_i = n - k
$$

求一个简单近似解：当 $v_{i}$ 两两正交时，有：

$$
\left\Vert\sum_{i=1}^n \gamma_i \boldsymbol{v}_i\right\Vert^2 = \sum_{i=1}^n \gamma_i^2 \Vert\boldsymbol{v}_i\Vert^2 = \sum_{i=1}^n \gamma_i \Vert\boldsymbol{v}_i\Vert^2
$$

上式最优解显然就是让模长 $\Vert\boldsymbol{v}_i\Vert$ 最小的 $n-k$ 个 $\gamma_{i}$ 等于 1，这又等价于说挑出模长最大的 $k$ 个向量来逼近 $n$ 个向量之和。它的几何意义也很直观，模长越大的向量，在求和过程中越不容易被抵消，从而作用越突出。
挑模长最大的 $k$ 个向量需要首先计算所有 $v_{i}$，这并不满足降低计算的需求，因此需要重新设计每个 Expert 模型。首先将 $v_{i}$ 归一化 $\boldsymbol{e}_i = \boldsymbol{v}_i/\Vert\boldsymbol{v}_i\Vert$ 统一模长，接着定义：

$$
\underbrace{[p_1,p_2,\cdots,p_n]}_{\boldsymbol{p}} = h(\boldsymbol{x}\cdot\boldsymbol{W}^{(R)})\quad\in\mathbb{R}_{\geq 0}^n
$$

其中 $\boldsymbol{W}^{(R)}\in\mathbb{R}^{d\times n}$ 是参数矩阵，$h(\cdot)$ 是一个 $\mathbb{R}\to\mathbb{R}_{\geq 0}$ 的激活函数，简单来说就是一个 $d$ 维到 $n$ 维的线性变换加激活函数，将 $p_{i}$ 作为第 $i$ 个 Expert 的模长，此时 Expert 分解为两个部分 $p_{i}e_{i}$：计算量比较小的模长 $p_{i}$ 以及计算量比较大的方向 $e_{i}$。为了减少计算量，先计算出 $p$，再挑出最大的 $k$ 个后再计算相应的 $e_{i}$，最后乘上 $p_{i}$ 并求和：

$$
\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{p}} p_i \boldsymbol{e}_i
$$

没有去考虑 Router 如何选择 Expert，只是每一步都尽可能逼近 Dense 模型，这可以说是**既要**大参数、**又要**小计算量的最佳选择。
