---
tags:
  - Flow
aliases: NICE
title: None-Linear Independent Components Estimation
date: 2026-01-16T05:37:34.685Z
draft: false
---
NICE is a deep learning framework changing high dimensional complex data into non linear independent components.
由独立成分分析（ICA）推广而来，使用非线性函数 $f$ 来生成观测数据，旨在将观察到的信号分离成许多潜在信号，潜在信号通过缩放和叠加可以恢复成观测数据，并且这些潜在信号是**完全独立**的。
- *Assume:* A good representation is one in which the data has a distribution that is easy to model. 
- *Purpose:* Changing high dimensional complex data into non linear independent components.
- Method：

$$
P_{H}~(h) = \prod_{d \in \mathcal{D}}~P_{H_{d}}~(h_{d})
$$


$$
P_{X}~(x) = \underbrace{P_{H}~(f(x))}_\text{prior distribution} ~ \left \vert \det \frac{\partial f(x)}{\partial x} \right \vert
$$


$$
\log( P_{X}~(x)) = \sum_{d=1}^{\mathcal{D}} \log(P_{H_{d}}~(f_{d}(x))) + \log\left(\left \vert \det \frac{\partial f(x)}{\partial x} \right \vert \right)
$$

---
### Additive Coupling 加性耦合
将 D 维的 x 分为两部分，再使用巧妙的变换：
$$
\begin{array}{l}
x = (x_{1}, x_{2}), ~ h = (h_{1}, h_{2}), ~ \text{m is a any function} \\
h_{1} = x_{1} \\
h_{2} = x_{2} + m~(x_{1}) \\
\end{array}
$$
- 变换可逆，并且产生单位常数行列式。
$$
J(h) = \begin{bmatrix}  \frac{ \partial h_{1}}{\partial x_{1} } & \frac{ \partial h_{1}}{\partial x_{2} }\\ \frac{ \partial h_{2}}{\partial x_{1} } & \frac{ \partial h_{2}}{\partial x_{2} } \end{bmatrix} = \begin{bmatrix} I_{d} & 0 \\ \frac{\partial m~(x_{1})}{\partial x_{1}}  & I_{d} \end{bmatrix}
$$
- 复合多个上述变换即可达到可观看的非线性表现， 即为加性耦合层的耦合，称为`flow`。根据链式法则，此时任然产生单位常数行列式。
$$
\begin{array}{l}
h_{1} = x_{1} \\
h_{2} = x_{2} + m_{1}~(x_{1}) \\
h_{3} = h_{1} + m_{2}~(x_{2}) \\
h_{4} = h_{2} \\
\end{array}
$$
> 为得到不平凡的变换，耦合顺序交错

$$
J(h) = \begin{bmatrix}  \frac{ \partial h_{3}}{\partial h_{1} } & \frac{ \partial h_{3}}{\partial h_{2} }\\ \frac{ \partial h_{4}}{\partial h_{1} } & \frac{ \partial h_{4}}{\partial h_{2} } \end{bmatrix} ~ \begin{bmatrix}  \frac{ \partial h_{1}}{\partial x_{1} } & \frac{ \partial h_{1}}{\partial x_{2} }\\ \frac{ \partial h_{2}}{\partial x_{1} } & \frac{ \partial h_{2}}{\partial x_{2} } \end{bmatrix} = \begin{bmatrix} I_{d} & \frac{\partial m_{2}~(x_{2})}{ \partial h_{2}} \\ 0 & I_{d} \end{bmatrix} \begin{bmatrix} I_{d} & 0 \\ \frac{\partial m_{1}~(x_{1})}{\partial x_{1}}  & I_{d} \end{bmatrix}
$$


![链式规则在多变量函数中是如何工作的](https://www.onurtunali.com/img/nice/der_depend.jpg)

---
### Scaling Layer 尺度变换

- 基于可逆变换模型存在内在维度浪费问题：变换函数（编码）分布在 D 维空间，但测试样本（生成）并不是 D 维流形。
$$
h = \exp(s) \odot h
$$
对编码出的每个维度做尺度变换（可优化的参数向量），提取重要的维度信息，压缩流形。

> 尺度变换层等价于将先验分布的方差（标准差）也作为训练参数，如果方差足够小，我们就可以认为该维度所表示的流形坍缩为一个点，从而总体流形的维度减 1，暗含了降维的可能。
