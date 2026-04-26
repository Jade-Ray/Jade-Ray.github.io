---
tags:
  - Transformers
  - Attention
aliases: null
title: Position Embedding
date: 2025-02-11T12:21:50.576Z
draft: false
---
位置编码是 Transformer 模型中独有的问题，由于纯粹的 Attention 机制与序列输入顺序无关，注意力运算有全对称性 $f(x,y) = f(y,x)$，模型无法根据结果区分对称位置上的 Token 输入。因此，能分辨位置信息的位置编码方法变得不可或缺，无论方法如何变化，其底层逻辑要满足以下方面：
- 编码方式简单；
- 位置编码可以外推，尽可能不受序列长度影响；
## 绝对位置编码
作为最简单的一种方法，直接对第 $k$ 个向量 $x_{k}$ 中加入位置向量 $p_{k}$ 变为 $x_{k} + p_{k}$，其中 $p_{k}$ 只依赖于位置编号 $k$，假设模型为 $f(\cdots,\boldsymbol{x}_{m},\cdots,\boldsymbol{x}_{n},\cdots)$，添加位置编码后为：

$$
\tilde{f}(\cdots,\boldsymbol{x}_{m},\cdots,\boldsymbol{x}_{n},\cdots) = f(\cdots,\boldsymbol{x}_{m} + \boldsymbol{p}_{m},\cdots,\boldsymbol{x}_{n} + \boldsymbol{p}_{n},\cdots)
$$

简单分析 $m,n$ 两个位置上的位置编码，将添加的位置编码视为扰动项，泰勒展开到二阶：

$$
\tilde{f}\approx f + \boldsymbol{p}_m^{\top} \frac{\partial f}{\partial \boldsymbol{x}_m} + \boldsymbol{p}_n^{\top} \frac{\partial f}{\partial \boldsymbol{x}_n} + \frac{1}{2}\boldsymbol{p}_m^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_m^2}\boldsymbol{p}_m + \frac{1}{2}\boldsymbol{p}_n^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_n^2}\boldsymbol{p}_n + \underbrace{\boldsymbol{p}_m^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_m \partial \boldsymbol{x}_n}\boldsymbol{p}_n}_{\boldsymbol{p}_m^{\top} \boldsymbol{\mathcal{H}} \boldsymbol{p}_n}
$$

其中第一项与位置无关，第二到五项只依赖单一位置（绝对位置信息），第六项是第一个同时包含 $p_{m},p_{n}$ 的交互项，记作 $\boldsymbol{p}_m^{\top} \boldsymbol{\mathcal{H}} \boldsymbol{p}_n$，我们希望研究其是否表达一定的相对位置信息。最简单的情况，假设 $\boldsymbol{\mathcal{H}} = \boldsymbol{I}$ 是单位矩阵，此时该交互项 $\boldsymbol{p}_m^{\top} \boldsymbol{\mathcal{H}} \boldsymbol{p}_n = \boldsymbol{p}_m^{\top} \boldsymbol{p}_n = \langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle$ 可视为两个位置的内积，若其表达的是相对位置信息，即存在某个函数 $g$ 使得：

$$
\langle\boldsymbol{p}_{m}, \boldsymbol{p}_{n}\rangle = g(m - n)
$$

假设位置编码向量为二维，并借助复数来推导，两个复数位置的内积为 $\langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle = \text{Re}[\boldsymbol{p}_m \boldsymbol{p}_n^*]$，其中 $\boldsymbol{p}_n^*$ 是 $\boldsymbol{p}_{n}$ 的共轭复数，$\text{Re}[]$ 代表复数的实部，假设存在复数 $q_{m-n}$ 对应相对位置信息，借助欧拉公式转为指数形式 $\boldsymbol{p}_m=r_m e^{\text{i}\phi_m}, \boldsymbol{p}_n^*=r_n e^{-\text{i}\phi_n}, \boldsymbol{q}_{m-n}=R_{m-n} e^{\text{i}\Phi_{m-n}}$ 可得：

$$
r_{m} r_{n} e^{\text{i}(\phi_{m} - \phi_{n})} = R_{m-n} e^{\text{i}\Phi_{m-n}} 
\quad\Rightarrow\quad 
\left\{\begin{aligned}
&r_{m} r_{n} = R_{m-n}\\ 
&\phi_{m} - \phi_{n}=\Phi_{m-n}
\end{aligned}\right.
$$

对于实部，当 $m=n$ 时 $r_{m}^{2} = R_{0}$，即 $r_{m}$ 是一个常数，简单设为 1；对于虚部，当 $n=0$ 时 $\phi_{m} - \phi_{0} = \Phi_{m}$，假设 $\phi_{0} = 0$，那么 $\phi_{m} = \Phi_{m}$，即 $\phi_{m} - \phi_{n} = \phi_{m-n}$，代入 $n = m-1$ 得 $\phi_{m}- \phi_{m-1} = \phi_{1}$，那么 $\{\phi_{m}\}$ 为等差数列，通解为 $m\theta$，则二维情况下得位置编码的一个显示解为：

$$
\boldsymbol{p}_m = e^{\text{i}m\theta}\quad\Leftrightarrow\quad \boldsymbol{p}_m=\begin{pmatrix}\cos m\theta \\ \sin m\theta\end{pmatrix}
$$

由于内积满足线性叠加性，所以更高维的位置编码，我们可以表示为多个二维位置编码的组合：

$$
\boldsymbol{p}_m = \begin{pmatrix}e^{\text{i}m\theta_0} \\ e^{\text{i}m\theta_1} \\ \vdots \\ e^{\text{i}m\theta_{d/2-1}}\end{pmatrix}\quad\Leftrightarrow\quad \boldsymbol{p}_m=\begin{pmatrix}\cos m\theta_0 \\ \sin m\theta_0 \\ \cos m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \sin m\theta_{d/2-1}  \end{pmatrix}
$$

总之，如何对绝对位置编号进行编码有多种模式。
### 训练式
直接将位置编码当作可训练的参数矩阵 $L_{max} \times D_{emb}$，其中 $L_{max}$ 是序列最大长度，$D_{emb}$ 是编码维度，随机初始化，随着训练过程进行更新。该方法一个明显的问题是无法处理超出最大长度的序列，但其编码效果更好。
### 三角式
利用三角函数的和差角公式设计的具有一定外推性的 Sinusoidal 位置编码：

$$
\begin{equation}\left\{\begin{aligned}
&\boldsymbol{p}_{k,2i}=\sin\Big(k/10000^{2i/d}\Big)\\
&\boldsymbol{p}_{k, 2i+1}=\cos\Big(k/10000^{2i/d}\Big) \end{aligned}\right.
\end{equation}
$$

其中 $\boldsymbol{p}_{k,2i},\boldsymbol{p}_{k,2i+1}$ 分别是位置 $k$ 的编码向量的第 $2i,2i+1$ 个分量，$d$ 是位置向量的维度。
根据和差角公式 $\sin(\alpha+\beta)=\sin\alpha\cos\beta+\cos\alpha\sin\beta$ 和 $\cos(\alpha+\beta)=\cos\alpha\cos\beta-\sin\alpha\sin\beta$ 可以看出，位置 $\alpha + \beta$ 的向量可以表示成位置 $\alpha$ 和位置 $\beta$ 的向量组合，提供了表达相对位置的可能性。
此外， Sinusoidal 位置编码可以看作上文相对位置信息项的一个解 $\theta_{i} = 10000^{-2i/d}$，随着编码位置的增加，位置编码结果以三角函数形式周期波动，这个形式有个特点，随着 $|m - n|$ 的增大，$\langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle$ 趋于零衰减，符合相对距离越大，相关性越弱的直觉假设，具体来说，内积可以表示为：

$$
\begin{aligned} 
\langle\boldsymbol{p}_{m}, \boldsymbol{p}_{n}\rangle 
=&\, \text{Re}\left[e^{\text{i}(m-n)\theta_0} + e^{\text{i}(m-n)\theta_1} + \cdots + e^{\text{i}(m-n)\theta_{d/2-1}}\right]\\ 
=&\,\frac{d}{2}\cdot\text{Re}\left[\sum_{i=0}^{d/2-1} e^{\text{i}(m-n)10000^{-i/(d/2)}}\frac{1}{d/2}\right]\\ 
\sim&\, \frac{d}{2}\cdot\text{Re}\left[\int_0^1 e^{\text{i}(m-n)\cdot 10000^{-t}}dt\right]
\end{aligned}
$$

震荡积分 $\int_0^1 e^{\text{i}(m-n)\theta_t}dt$ 具有渐进衰减趋势。
![](https://spaces.ac.cn/usr/uploads/2021/03/300971803.png)
## 相对位置编码
从上面绝对位置编码的介绍可以看出，无需完整建模每个输入的位置信息，只考虑当前位置与需要注意力的位置的相对距离，即可实现位置编码，称为相对位置编码，其更灵活，且不受长度限制。
### 经典式
相对位置编码起源于 Google 的论文[《Self-Attention with Relative Position Representations》](https://papers.cool/arxiv/1803.02155)，先从绝对位置的注意力机制开始讨论，考虑一般的形式：

$$
\left\{\begin{aligned} 
\boldsymbol{q}_i =&\, (\boldsymbol{x}_i + \boldsymbol{p}_i)\boldsymbol{W}_Q \\ 
\boldsymbol{k}_j =&\, (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_K \\ 
\boldsymbol{v}_j =&\, (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_V \\ 
a_{i,j} =&\, softmax\left(\boldsymbol{q}_i \boldsymbol{k}_j^{\top}\right)\\ 
\boldsymbol{o}_i =&\, \sum_j a_{i,j}\boldsymbol{v}_j 
\end{aligned}\right.
$$

其中 softmax 对 $j$ 那一维归一化，展开 $q_{i}k_{i}^{T}$：

$$
\boldsymbol{q}_i \boldsymbol{k}_j^{\top} = \left(\boldsymbol{x}_i + \boldsymbol{p}_i\right)\boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\left(\boldsymbol{x}_j + \boldsymbol{p}_j\right)^{\top} = \left(\boldsymbol{x}_i \boldsymbol{W}_Q + \boldsymbol{p}_i \boldsymbol{W}_Q\right)\left(\boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{W}_K^{\top}\boldsymbol{p}_j^{\top}\right)
$$

不考虑 $i$ 上的查询位置向量 $p_{i}$，简化得 $q_{i}k_{j}^{T} = x_{i}W_{Q}(x_{j}W_{K} + p_{j}W_{K})^{T}$，将 $j$ 上的键位置向量 $p_{j}W_{K}$ 改为与 $i,j$ 有关的二元位置向量 $R_{i,j}^{K}$，变成：

$$
a_{i,j} = softmax\left(\boldsymbol{x}_i \boldsymbol{W}_Q\left(\boldsymbol{x}_j\boldsymbol{W}_K + {\color{green}{\boldsymbol{R}_{i,j}^{K}}}\right)^{\top}\right)
$$

并将 $j$ 上的值位置向量 $p_{j}W_{V}$ 也改为与 $i,j$ 有关的二元位置向量 $R_{i,j}^{V}$，变成：

$$
\boldsymbol{o}_i = \sum_j a_{i,j}\left(\boldsymbol{x}_j\boldsymbol{W}_V + {\color{green}{\boldsymbol{R}_{i,j}^{V}}}\right)
$$

对于依赖二元坐标 $(i,j)$ 的向量 $R_{i,j}^{K},R_{i,j}^{V}$，可以改为依赖相对距离 $i-j$ 的相对位置向量，并添加截断以适应任意距离：

$$
\begin{aligned} 
\boldsymbol{R}_{i,j}^{K} = \boldsymbol{p}_K\left[\text{clip}(i-j, p_{\min}, p_{\max})\right]\\ 
\boldsymbol{R}_{i,j}^{V} = \boldsymbol{p}_V\left[\text{clip}(i-j, p_{\min}, p_{\max})\right] 
\end{aligned}
$$

此时，只需有限个位置编码，即可表达任意长度的相对位置，$p_{K},p_{V}$ 选择训练式还是三角函数式都可。
### XLNET 式
XLNET 式位置编码源自 Transformer-XL 的论文[《Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context》](https://papers.cool/arxiv/1901.02860)，该位置编码源于对上述 $q_{i}k_{j}^{T}$ 的完全展开：

$$
\boldsymbol{q}_i \boldsymbol{k}_j^{\top} = \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{p}_j^{\top} + \boldsymbol{p}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{p}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{p}_j^{\top}
$$

直接将 $p_{j}$ 替换为相对位置向量 $R_{i-j}$，两个 $p_{i}$ 替换为可训练向量 $u,v$：

$$
\boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}{\color{green}{\boldsymbol{R}_{i-j}^{\top}}} +  {\color{red}{\boldsymbol{u}}}\boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + {\color{red}{\boldsymbol{v}}} \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\color{green}{\boldsymbol{R}_{i-j}^{\top}}
$$

式中 $R_{i-j}$ 不使用截断，而是用 Sinusoidal 方式生成，并且由于 $R_{i-j}$ 的编码空间不一定与 $x_{j}$ 相同，所以 $W_{K}^{T}$ 换成独立矩阵 $W_{K,R}^{T}$，同时 ${\color{red}{u}}W_{Q}$、 ${\color{red}{v}}W_{Q}$ 可以简化为单个的 ${\color{red}{u}}$、 ${\color{red}{v}}$：

$$
\boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_{K,R}^{\top}{\color{green}{\boldsymbol{R}_{i-j}^{\top}}} +  {\color{red}{\boldsymbol{u}}}\boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + {\color{red}{\boldsymbol{v}}} \boldsymbol{W}_{K,R}^{\top}{\color{green}{\boldsymbol{R}_{i-j}^{\top}}}
$$

此外，$v_{j}$ 上不再加位置偏置，即 $\boldsymbol{o}_i = \sum\limits_j a_{i,j}\boldsymbol{x}_j\boldsymbol{W}_V$。
### T5 式
T5 模型出自文章[《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》](https://papers.cool/arxiv/1910.10683)，里边用到了一种更简单的相对位置编码。该方法认为对于 $q_{i}k_{j}^{T}$ 完全展开的四项，可以看作为“输入-输入”、“输入-位置”、“位置-输入”、“位置-位置”四项注意力的组合。若认为输入信息应与位置信息解耦（位置与输入是相互独立的），那么只用保留输入交互项和位置交互项 $\boldsymbol{p}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{p}_j^{\top}$，并且可以直接将该位置交互项作为参数训练出来：

$$
\boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \color{green}{\boldsymbol{\beta}_{i,j}}
$$

简单来说，该方法相对于仅在 Attention 矩阵上加了一个可训练的偏置项。
### DeBERTa 式
出自论文[《DeBERTa: Decoding-enhanced BERT with Disentangled Attention》](https://papers.cool/arxiv/2006.03654)，与 T5 式相反，该方法去除了 $q_{i}k_{j}^{T}$ 完全展开的第四项，保留前三项：

$$
\boldsymbol{q}_i \boldsymbol{k}_j^{\top} = \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}{\color{green}{\boldsymbol{R}_{i,j}^{\top}}} + {\color{green}{\boldsymbol{R}_{j,i}}} \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top}
$$

其中 $R_{i,j},R_{i,j}$ 的设置与经典式一致。此外，该方法认为相对位置信息和绝对位置信息同样重要，在 13 层的 BERT 中，在前 11 层只使用相对位置编码，最后两层中再加入绝对位置信息。
### 旋转式位置编码 （RoPE）
通过绝对位置编码的方式实现相对位置编码，这样可以同时具备两者的优点，且具备优雅的数学表示形式。
首先，通过对 $q,k$ 添加绝对位置信息 $m,n$，我们希望 Attention 运算（内积）的结果带有相对位置信息，那么便要得到下面恒等式的一个简单解：

$$
\langle\boldsymbol{f}(\boldsymbol{q}, m), \boldsymbol{f}(\boldsymbol{k}, n)\rangle = g(\boldsymbol{q},\boldsymbol{k},m-n)
$$

同[绝对位置编码](#绝对位置编码)中讨论的思路，先借助复数考虑二维情形，内积 $\langle\boldsymbol{f}(\boldsymbol{q}, m), \boldsymbol{f}(\boldsymbol{k}, n)\rangle$ 变为实部的乘法 $\text{Re}[\boldsymbol{f}(q,m) \boldsymbol{f}^*(k,n)]$，再根据欧拉公式转为指数形式：

$$
\begin{aligned} 
\boldsymbol{f}(\boldsymbol{q}, m) =&\, R_f (\boldsymbol{q}, m)e^{\text{i}\Theta_f(\boldsymbol{q}, m)} \\ 
\boldsymbol{f}(\boldsymbol{k}, n) =&\, R_f (\boldsymbol{k}, n)e^{\text{i}\Theta_f(\boldsymbol{k}, n)} \\ 
\boldsymbol{g}(\boldsymbol{q}, \boldsymbol{k}, m-n) =&\, R_g (\boldsymbol{q}, \boldsymbol{k}, m-n)e^{\text{i}\Theta_g(\boldsymbol{q}, \boldsymbol{k}, m-n)} \\ 
\end{aligned}
$$

带入等式可得方程组：

$$
\begin{aligned} 
R_f (\boldsymbol{q}, m) R_f (\boldsymbol{k}, n) =&\, R_g (\boldsymbol{q}, \boldsymbol{k}, m-n) \\ 
\Theta_f (\boldsymbol{q}, m) - \Theta_f (\boldsymbol{k}, n) =&\, \Theta_g (\boldsymbol{q}, \boldsymbol{k}, m-n) 
\end{aligned}
$$

前一个方程，带入 $m=n$ 得：

$$
R_f (\boldsymbol{q}, m) R_f (\boldsymbol{k}, m) = R_g (\boldsymbol{q}, \boldsymbol{k}, 0) = R_f (\boldsymbol{q}, 0) R_f (\boldsymbol{k}, 0) = \Vert \boldsymbol{q}\Vert \Vert \boldsymbol{k}\Vert
$$

其初始条件有 $f(q,0)=q$ 和 $f(k,0)=k$，此时可同样设 $R_f (\boldsymbol{q}, m) = \Vert \boldsymbol{q}\Vert, R_f (\boldsymbol{k}, m) = \Vert \boldsymbol{k}\Vert$，即不依赖于 $m$。再将 $m=n$ 带入第二个方程：

$$
\Theta_f (\boldsymbol{q}, m) - \Theta_f (\boldsymbol{k}, m) = \Theta_g (\boldsymbol{q}, \boldsymbol{k}, 0) = \Theta_f (\boldsymbol{q}, 0) - \Theta_f (\boldsymbol{k}, 0) =  \Theta (\boldsymbol{q}) - \Theta (\boldsymbol{k})
$$

这里虚部 $\Theta(q), \Theta(k)$ 是 $q,k$ 的幅角，将上式变换为 $\Theta_f (\boldsymbol{q}, m) - \Theta (\boldsymbol{q}) = \Theta_f (\boldsymbol{k}, m) - \Theta (\boldsymbol{k})$，其中 $\Theta_f (\boldsymbol{q}, m) - \Theta (\boldsymbol{q})$ 看作一个只与 $m$ 有关、跟 $q$ 无关的函数 $\varphi(m)$，即 $\Theta_f (\boldsymbol{q}, m) = \Theta (\boldsymbol{q}) + \varphi(m)$，接着带入 $n=m-1$：

$$
\varphi(m) - \varphi(m-1) = \Theta_g (\boldsymbol{q}, \boldsymbol{k}, 1) + \Theta (\boldsymbol{k}) - \Theta (\boldsymbol{q})
$$

即 $\{\varphi(m)\}$ 为等差数列，设右端为 $\theta$，那么恒等式的一个解为 $\varphi(m) = m\theta$，带入复数内积的指数形式中：

$$
\boldsymbol{f}(\boldsymbol{q}, m) = R_f (\boldsymbol{q}, m)e^{\text{i}\Theta_f(\boldsymbol{q}, m)} = \Vert q\Vert e^{\text{i}(\Theta(\boldsymbol{q}) + m\theta)} = \boldsymbol{q} e^{\text{i}m\theta}
$$

欧拉公式表明复平面的乘法等价为几何上向量的旋转，对于二维向量 $q[q_{0},q_{1}]$，其指数形式展开为 $(q_{0}+q_{1}i)e^{im\theta} = (q_{0}+q_{1}i)(\cos m\theta + i \sin m\theta) = (q_{0} \cos m\theta - q_{1} \sin m\theta) + i(q_{0} \sin m\theta + q_{1} \cos m\theta)$ 用矩阵形式表示：

$$
\boldsymbol{f}(\boldsymbol{q}, m) =\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix} \begin{pmatrix}q_0 \\ q_1\end{pmatrix}
$$

由于内积满足线性叠加性，因此任意偶数维的 RoPE，我们都可以表示为二维情形的拼接，即:

$$
\scriptsize{\underbrace{\begin{pmatrix} \cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\  \sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\ 0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\ 0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \\ \end{pmatrix}}_{\boldsymbol{\mathcal{R}}_m} \begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1}\end{pmatrix}}
$$

也就是说，给位置 $m$ 的向量 $q$ 乘以矩阵 $\boldsymbol{\mathcal{R}}_m$、位置 $n$ 的向量 $k$ 乘以矩阵 $\boldsymbol{\mathcal{R}}_n$，用变换后的$Q,K$ 序列进行 Attention 计算，结果就自动包含相对位置信息了，因为成立恒等式：

$$
(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) =  \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n \boldsymbol{k} = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}
$$

值得指出的是 $\boldsymbol{\mathcal{R}}_m$ 作为一个正交矩阵，其不会改变向量的模长，因此通常来说它不会改变原模型的稳定性。
考虑到 $\boldsymbol{\mathcal{R}}_m$ 的稀疏性，采用 $\otimes$ 进行逐位相乘更高效：

$$
\begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} \end{pmatrix}\otimes
\begin{pmatrix}\cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \cos m\theta_{d/2-1} \end{pmatrix} + 
\begin{pmatrix}-q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \\ -q_{d-1} \\ q_{d-2} \end{pmatrix}\otimes
\begin{pmatrix}\sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1} \end{pmatrix}
$$

与 [三角式](#三角式) 的形式类似，RoPE 同样具备远程衰减性。此外，该方法不直接基于 Attention 矩阵进行操作，可以应用到线性 Attention 中。
#### 二维 RoPE
对于二维 $h\times h$ 的 feature map，简单展开成一维处理是不恰当的。考虑位置 $(x,y)$ 展平后为 $xh+y$，位置 $(x+1,y)$ 和 $(x,y+1)$ 展平后分别为 $xh+y+h$ 和 $xh + y + 1$，两者在二维平面上与 $(x,y)$ 的距离是相同的，但在展平后的一维上与 $xh + y$ 的距离却不同。
因此，二维情形需要重新设计，回顾一维 RoPE 的矩阵 $\boldsymbol{\mathcal{R}}_n=\begin{pmatrix}\cos n\theta & -\sin n\theta\\ \sin n\theta & \cos n\theta\end{pmatrix}$，其满足“相对性”条件：

$$
\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n=\boldsymbol{\mathcal{R}}_{n-m}
$$

可以假设，对于求解的二维矩阵 $\boldsymbol{\mathcal{R}}_{x,y}$，同样要满足二维的相对性条件 $\boldsymbol{\mathcal{R}}_{x_1,y_1}^{\top}\boldsymbol{\mathcal{R}}_{x_2,y_2}=\boldsymbol{\mathcal{R}}_{x_2-x_1,y_2-y_1}$，同时为了无损编码图像相对位置，应确保该矩阵满足“可逆性”，即给定 $\boldsymbol{\mathcal{R}}_{x,y}$ 可以反解出 $x,y$。
经过一番推导，得到二维 RoPE 的一个解为：

$$
\boldsymbol{\mathcal{R}}_{x,y}=\left( 
\begin{array}{cc:cc} 
\cos x\theta & -\sin x\theta & 0 & 0 \\ 
\sin x\theta & \cos x\theta & 0 & 0 \\ 
\hdashline 
0 & 0 & \cos y\theta & -\sin y\theta \\ 
0 & 0 & \sin y\theta & \cos y\theta \\ 
\end{array}\right)
= \begin{pmatrix}\boldsymbol{\mathcal{R}}_x & 0 \\ 0 & \boldsymbol{\mathcal{R}}_y\end{pmatrix}
$$

##### 矩阵指数推导过程
这里的[矩阵指数](https://en.wikipedia.org/wiki/Matrix_exponential)是按照幂级数定义的运算：

$$
\exp \boldsymbol{B} = \sum_{k=0}^{\infty}\frac{\boldsymbol{B}^k}{k!}
$$

一维的 RoPE 存在比较简单的指数表达式：

$$
\boldsymbol{\mathcal{R}}_n=\begin{pmatrix}\cos n\theta & -\sin n\theta\\ \sin n\theta & \cos n\theta\end{pmatrix}=\exp\left\{n\theta\begin{pmatrix}0 & -1\\ 1 & 0\end{pmatrix}\right\}
$$

假设 RoPE 的解为 $\boldsymbol{\mathcal{R}}_n=\exp n\boldsymbol{B}$，其中 $\boldsymbol{B}$ 是一个跟 $n$ 无关的矩阵。因为解 $\boldsymbol{\mathcal{R}}_n$ 需要满足“相对性”条件，故有：

$$
\big(\exp m\boldsymbol{B}\big)^{\top}\big(\exp n\boldsymbol{B}\big) = \big(\exp m\boldsymbol{B}^{\top}\big)\big(\exp n\boldsymbol{B}\big)
$$

根据矩阵指数存在的性质：

$$
\boldsymbol{A}\boldsymbol{B} = \boldsymbol{B}\boldsymbol{A} \quad\Rightarrow\quad \big(\exp \boldsymbol{A}\big)\big(\exp \boldsymbol{B}\big) = \exp \big(\boldsymbol{A} + \boldsymbol{B}\big)
$$

假设 $\boldsymbol{B}^{T},\boldsymbol{B}$ 可交换，则：

$$
\big(\exp m\boldsymbol{B}^{\top}\big)\big(\exp n\boldsymbol{B}\big) = \exp \big(m\boldsymbol{B}^{\top} + n\boldsymbol{B}\big)
$$

要让 $m\boldsymbol{B}^{\top} + n\boldsymbol{B}=(n-m)\boldsymbol{B}$，只须满足 $\boldsymbol{B}^{\top} = - \boldsymbol{B}$，即 $\boldsymbol{B}$ 为正交矩阵，对于 $2\times 2$ 的矩阵来说，$\boldsymbol{B}^{\top} + \boldsymbol{B} = 0$ 的通解是 $\boldsymbol{B}=\begin{pmatrix}0 & -\theta\\ \theta & 0\end{pmatrix}$。
类似地，二维 RoPE 中有：

$$
\boldsymbol{\mathcal{R}}_{x,y}=\exp \big(x\boldsymbol{B}_1 + y\boldsymbol{B}_2\big)
$$

假设 $x_1\boldsymbol{B}_1^{\top} + y_1\boldsymbol{B}_2^{\top}$ 与 $x_2\boldsymbol{B}_1 + y_2\boldsymbol{B}_2$ 可交换，有：

$$
\big(\exp (x_{1}\boldsymbol{B}_{1} + y_{1}\boldsymbol{B}_{2})\big)^{\top} \big(\exp (x_{2}\boldsymbol{B}_{1} + y_{2}\boldsymbol{B}_{2})\big) = \exp \big((x_1\boldsymbol{B}_1^{\top} + y_1\boldsymbol{B}_2^{\top}) + (x_2\boldsymbol{B}_1 + y_2\boldsymbol{B}_2)\big)
$$

那么可以得到以下约束条件：

$$
\left\{\begin{aligned} 
&\boldsymbol{B}_1^{\top} + \boldsymbol{B}_1 = 0\\ 
&\boldsymbol{B}_2^{\top} + \boldsymbol{B}_2 = 0\\ 
&\boldsymbol{B}_1 \boldsymbol{B}_2^{\top} = \boldsymbol{B}_2^{\top} \boldsymbol{B}_1 
\end{aligned}\right.
$$

若解 $\boldsymbol{B}_{1},\boldsymbol{B}_{2}$ 为 $2 \times 2$ 矩阵，其只有一个独立参数不满足“可逆性”。考虑为 $3 \times 3$ 矩阵，存在 3 个独立参数，为了保证可逆性，不妨设 $\boldsymbol{B}_{1},\boldsymbol{B}_{2}$ 是正交的：

$$
\boldsymbol{B}_1=\begin{pmatrix}0 & -a & 0 \\ a & 0 & 0 \\ 0 & 0 & 0\end{pmatrix},\quad\boldsymbol{B}_2=\begin{pmatrix}0 & 0 & -b \\ 0 & 0 & -c \\ b & c & 0\end{pmatrix}
$$

当 $a=1$ 时，可解得 $b=0,c=0$，零解不是我们希望得到的解。
考虑 $4\times 4$ 矩阵，存在 6 个独立参数，考虑其正交分解：

$$
\boldsymbol{B}_1=\begin{pmatrix}0 & -a & -b & 0 \\ a & 0 & -c & 0 \\ b & c & 0 & 0 \\ 0 & 0 & 0 & 0\end{pmatrix},\quad\boldsymbol{B}_2=\begin{pmatrix}0 & 0 & 0 & -d \\ 0 & 0 & 0 & -e \\ 0 & 0 & 0 & -f \\ d & e & f & 0\end{pmatrix}
$$

根据约束条件 $\boldsymbol{B}_1 \boldsymbol{B}_2 = \boldsymbol{B}_2 \boldsymbol{B}_1$ 可解得：

$$
d=cf,\quad e=-bf
$$

简单设 $a=1,f=1$，且其余参数为零，此时解为：

$$
\boldsymbol{\mathcal{R}}_{x,y}=\exp \,\begin{pmatrix}0 & -x & 0 & 0 \\ x & 0 & 0 & 0 \\ 0 & 0 & 0 & -y \\ 0 & 0 & y & 0\end{pmatrix}
$$

增加参数 $\theta$ 后完全展开可得：

$$
\boldsymbol{\mathcal{R}}_{x,y}
=\exp \,\left\{
\begin{pmatrix} 
0 & -x & 0 & 0 \\ 
x & 0 & 0 & 0 \\ 
0 & 0 & 0 & -y \\ 
0 & 0 & y & 0
\end{pmatrix}\theta\right\}
=\left( \begin{array}{cc:cc} 
\cos x\theta & -\sin x\theta & 0 & 0 \\ 
\sin x\theta & \cos x\theta & 0 & 0 \\ 
\hdashline 
0 & 0 & \cos y\theta & -\sin y\theta \\ 
0 & 0 & \sin y\theta & \cos y\theta \\ 
\end{array}\right)
$$

#### 多模态 RoPE
对于文本、视觉多模态输入，其位置编码所指并不相同。
![image.png](https://s2.loli.net/2025/02/11/4Sv2oOhgeZVfkd7.png)
![image.png](https://s2.loli.net/2025/02/11/hSP7CNAOyrWXwLa.png)
虽然将文本和图片都展平为一维来处理相当普遍，但降维后的图像位置编码随着图像尺寸的变大，其可表达的图像位置相关性也会随之减弱。
因此统一到二维空间位置坐标更为合理，根据[旋转式位置编码 （RoPE）](#旋转式位置编码-rope)和[二维 RoPE](#二维-rope)的结论：

$$
\scriptsize{\begin{array}{c}\begin{array}{c}\text{RoPE-1D}\\ (\boldsymbol{\mathcal{R}}_n)\end{array}= 
\begin{pmatrix} 
\cos {\color{yellow}{n}}\theta_0 & -\sin {\color{yellow}{n}}\theta_0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 \\ 
\sin {\color{yellow}{n}}\theta_0 & \cos {\color{yellow}{n}}\theta_0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 \\ 
0 & 0 & \cos {\color{yellow}{n}}\theta_1 & -\sin {\color{yellow}{n}}\theta_1 & \cdots & 0 & 0 & 0 & 0 \\ 
0 & 0 & \sin {\color{yellow}{n}}\theta_1 & \cos {\color{yellow}{n}}\theta_1 & \cdots & 0 & 0 & 0 & 0 \\ 
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots \\ 
0 & 0 & 0 & 0 & \cdots & \cos {\color{yellow}{n}}\theta_{d/2-2} & -\sin {\color{yellow}{n}}\theta_{d/2-2} & 0 & 0 \\ 
0 & 0 & 0 & 0 & \cdots & \sin {\color{yellow}{n}}\theta_{d/2-2} & \cos {\color{yellow}{n}}\theta_{d/2-2} & 0 & 0 \\ 
0 & 0 & 0 & 0 & \cdots & 0 & 0 & \cos {\color{yellow}{n}}\theta_{d/2-1} & -\sin {\color{yellow}{n}}\theta_{d/2-1} \\ 
0 & 0 & 0 & 0 & \cdots & 0 & 0 & \sin {\color{yellow}{n}}\theta_{d/2-1} & \cos {\color{yellow}{n}}\theta_{d/2-1} \\ 
\end{pmatrix} \\[16pt] 
\begin{array}{c}\text{RoPE-2D}\\ (\boldsymbol{\mathcal{R}}_{x,y})\end{array}= 
\begin{pmatrix} 
\cos {\color{yellow}{x}}\theta_0 & -\sin {\color{yellow}{x}}\theta_0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 \\ 
\sin {\color{yellow}{x}}\theta_0 & \cos {\color{yellow}{x}}\theta_0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 \\ 
0 & 0 & \cos {\color{yellow}{y}}\theta_1 & -\sin {\color{yellow}{y}}\theta_1 & \cdots & 0 & 0 & 0 & 0 \\ 
0 & 0 & \sin {\color{yellow}{y}}\theta_1 & \cos {\color{yellow}{y}}\theta_1 & \cdots & 0 & 0 & 0 & 0 \\ 
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots \\ 
0 & 0 & 0 & 0 & \cdots & \cos {\color{yellow}{x}}\theta_{d/2-2} & -\sin {\color{yellow}{x}}\theta_{d/2-2} & 0 & 0 \\ 
0 & 0 & 0 & 0 & \cdots & \sin {\color{yellow}{x}}\theta_{d/2-2} & \cos {\color{yellow}{x}}\theta_{d/2-2} & 0 & 0 \\ 
0 & 0 & 0 & 0 & \cdots & 0 & 0 & \cos {\color{yellow}{y}}\theta_{d/2-1} & -\sin {\color{yellow}{y}}\theta_{d/2-1} \\ 
0 & 0 & 0 & 0 & \cdots & 0 & 0 & \sin {\color{yellow}{y}}\theta_{d/2-1} & \cos {\color{yellow}{y}}\theta_{d/2-1} \\  \\ \end{pmatrix}\end{array}}
$$

$\boldsymbol{\mathcal{R}}_{x,y}$ 是 $\boldsymbol{\mathcal{R}}_{x}$ 和 $\boldsymbol{\mathcal{R}}_{y}$ 的分块对角组合，故 $\boldsymbol{\mathcal{R}}_{n,n}$ 可以看作是两个 $\boldsymbol{\mathcal{R}}_{n}$ 的分块对角组合，同时 RoPE-1D 的 $\boldsymbol{\mathcal{R}}_{n}^{(d\times d)}$ 是由多个不同 $\theta$ 的 $\boldsymbol{\mathcal{R}}_{n}$ 的分块对角组合。因此只要从 $\boldsymbol{\mathcal{R}}_{n}^{(d\times d)}$ 中选择不同 $\theta$ 给 $x,y$，那么 $\boldsymbol{\mathcal{R}}_{n,n}$ 就可以看成是 RoPE-1D （即 $\boldsymbol{\mathcal{R}}_{n}^{(d\times d)}$）的一部分。是故文本的位置应该采用 $(n, n)$ 的形式。
对于单张 $w \times h$ 个 patch 的图像来说，其二维位置坐标展平后为：

$$
\begin{array}{c|cccc|cccc|c|cccc} 
\hline 
x & 1 & 1 & \cdots & 1 & 2 & 2 & \cdots & 2 & \quad \cdots \quad & h & h & \cdots & h \\ 
\hline 
y & 1 & 2 & \cdots & w & 1 & 2 & \cdots & w & \quad \cdots \quad & 1 & 2 & \cdots & w \\ 
\hline 
\end{array}
$$

若该图片位于一个长度 $L$ 的句子后面，设定这个句子最后一个 token 的位置编码为 $(L,L)$，于是更新后的图片的位置编码为：

$$
\begin{array}{c|cccc|c|cccc} 
\hline 
x & L+1 & L+1 & \cdots & L+1 & \quad \cdots \quad & L+h & L+h & \cdots & L+h \\ 
\hline 
y & L+1 & L+2 & \cdots & L+w & \quad \cdots \quad & L+1 & L+2 & \cdots & L+w \\ 
\hline 
\end{array}
$$

但此时图片关于前后的句子存在不对称性，即图片前一个句子的最后一个 token 离图片的第一个 patch 的距离为 $(1,1)$，若图片后再接一个句子，设定该句子的第一个 token 的位置是 $(K,K)$，而图片的最后一个 patch 的位置则为 $(L+h,L+w)$，若 $w \neq h$， $(K,K)$ 和 $(L+h,L+w)$ 的距离不可能再保持为 $(1,1)$。为了保证对称性，将图片位置坐标进行缩放：

$$
\begin{array}{c|cccc|cccc|c|cccc} 
\hline 
x & s & s & \cdots & s & 2s & 2s & \cdots & 2s & \quad \cdots \quad & hs & hs & \cdots & hs \\ 
\hline 
y & t & 2t & \cdots & wt & t & 2t & \cdots & wt & \quad \cdots \quad & t & 2t & \cdots & wt \\ 
\hline 
\end{array}
$$

此时为了实现图片关于前后的句子的距离相同，应满足：

$$
\begin{pmatrix}L + hs \\ L + wt \end{pmatrix} + \begin{pmatrix}s \\ t \end{pmatrix} = \begin{pmatrix}K \\ K \end{pmatrix}\quad \Rightarrow \quad (h+1)s = (w+1)t
$$

考虑到 $h,w$ 的任意性，最简单的一个解为 $s = w+1,t=h+1$，新句子的第一个 token 的位置将会是 $K = L +(w+1)(h+1)$，如下所示（$s=5,t=4$）：
![|400](https://s2.loli.net/2025/02/11/xNamtOu3fvpQkHT.png)
虽然此时对称性得到满足，但文本 Token 和 Patch 并不具备一定的等价性，即是否可以将图片视作一个 $wh$ 个 Token 的句子，若图片左段文本的最后一个 Token 位置是 $(L,L)$，那么右端文本的第一个 Token 位置应该为 $(L+wh+1,L+wh+1)$，对于图像的一般二维位置：

$$
\left[\begin{matrix}  
(\beta_1 + \gamma_1,\beta_2 + \gamma_2) & (\beta_1 + \gamma_1,\beta_2 + 2\gamma_2) & \cdots & (\beta_1 + \gamma_1,\beta_2 + w\gamma_2) \\[8pt] 
(\beta_1 + 2\gamma_1,\beta_2 + \gamma_2) & (\beta_1 + 2\gamma_1,\beta_2 + 2\gamma_2) & \cdots & (\beta_1 + 2\gamma_1,\beta_2 + w\gamma_2) \\[8pt] 
\vdots & \vdots & \ddots & \vdots \\[8pt] 
(\beta_1 + h\gamma_1,\beta_2 + \gamma_2) & (\beta_1 + h\gamma_1,\beta_2 + 2\gamma_2) & \cdots & (\beta_1 + h\gamma_1,\beta_2 + w\gamma_2) 
\end{matrix}\right]
$$

应该有：

$$
\begin{pmatrix}\beta_1 + \gamma_1 \\ \beta_2 + \gamma_2\end{pmatrix} - \begin{pmatrix}L \\ L\end{pmatrix} = \begin{pmatrix}L+wh+1 \\ L+wh+1\end{pmatrix} - \begin{pmatrix}\beta_1 + h\gamma_1 \\ \beta_2 + w\gamma_2\end{pmatrix}
$$

简单取 $\gamma_1=\gamma_2=1$，解得：

$$
\beta_1 = L + \frac{1}{2}(wh - h),\quad \beta_2 = L + \frac{1}{2}(wh - w)
$$

新编码方式如下所示：
![|400](https://s2.loli.net/2025/02/11/8Jz4nCsF1vWVN36.png)
取 $\gamma_1=\gamma_2=1$ 有一个好处在于图像 Patch 间的间隔是固定的 $(0,1)$ 和 $(1,0)$，无论同一张图输入尺寸如何变化，相同位置、含义上的 Patch 间隔不受影响。
#### 三维混合模态位置编码
对于一个 $w\times h\times t$ 的视频（图像为 $w\times h$，同 $t$ 帧），它的位置坐标是三维的 $(x,y,z)$，根据相同的**兼容性**、**等价性**和**对称性**，可以推广到：

$$
\begin{pmatrix}\beta_1 + \gamma_1 \\ \beta_2 + \gamma_2 \\ \beta_3 + \gamma_3\end{pmatrix} - \begin{pmatrix}L \\ L \\ L\end{pmatrix} = \begin{pmatrix}L+wht+1 \\ L+wht+1 \\ L+wht+1\end{pmatrix} - \begin{pmatrix}\beta_1 + h\gamma_1 \\ \beta_2 + w\gamma_2 \\ \beta_3 + t\gamma_3\end{pmatrix}
$$

同样取 $\gamma_1=\gamma_2=\gamma_3=1$，解得：

$$
\beta_1 = L + \frac{1}{2}(wht - h),\quad \beta_2 = L + \frac{1}{2}(wht - w),\quad \beta_3 = L + \frac{1}{2}(wht - t)
$$

虽然得到了三维下的位置编码，但考虑到时间维度与图像空间维度构成的视频序列并不等价于空间三维度，时间维度的长度是不受限的，时间维度上的图像空间变化应该如同文本序列变化，其帧上的位置信息不应该受到总帧数的影响，一个可并行递归的时间维度帧编码方式更为合理。
