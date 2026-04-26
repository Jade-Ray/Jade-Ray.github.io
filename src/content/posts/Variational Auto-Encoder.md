---
tags:
  - VAE
aliases: VAE
title: Variational Auto-Encoder
date: 2026-04-20T16:49:51.884Z
draft: false
---
基于使用可微**生成器网络**，假设存在一个潜变量 $z$，模型使用可微函数 $g\left ( z ; \theta^{\left ( g \right )} \right)$ 将潜变量 $z$ 的样本变换为样本 $x$ 或样本 $x$ 上的分布（当然实现中潜变量 $z$ 假设为某些常见分布，方便从中采样生成图像）。

$$
p(X)=\sum_Z p(X|Z)p(Z)
$$

当假设 $Z$ 服从标准正态分布，从中采样$Z_1, Z_2, \dots, Z_n$，然后通过变换得到 $\hat{X}_1 = g(Z_1),\hat{X}_2 = g(Z_2),\dots,\hat{X}_n = g(Z_n)$，现在需要判断生成器$g$构造的数据是否与目标一致（此处无法使用 KL 散度，因为我们不知道概率分布的具体形式）。
- 对于 GAN，我们无法知道从正态分布$p(Z)$中采样的$Z$到底对应哪个真实数据采样$X$，因此直接用网络学习两者的度量。
- 对于 VAE，没有使用 $Z$ 服从标准正态分布假设，而是假设后验分布$p(X \mid Z)$为正态分布！$\log_{q_{\phi}}(z \mid x^{(i)}) = \log \mathcal{N}(z;\mu^{i}, \sigma^{2(i)} I)$此时我们可以通过对均值和方差建模（编码）学习真实数据$x^{(i)}$对应的后验分布，基于该分布采样的潜变量$z^{(i)}$经过生成器得到的$\hat{x}^{(i)}=g(z^{(i)})$便可以与真实值进行比较，即最小化$\mathcal{D}(\hat{x}^{(i)}, {x}^{(i)})$。 
 - 当所有$p(X \mid Z)$都向标准正态分布$\mathcal{N}(0,I)$看齐时，那么根据定义$p(Z)=\sum_X p(Z|X)p(X)=\sum_X \mathcal{N}(0,I)p(X)=\mathcal{N}(0,I) \sum_X p(X) = \mathcal{N}(0,I)$ 实现了我们的先验假设 $Z$ 服从标准正态分布，也就可以从$\mathcal{N}(0,I)$中采样生成图像了。
 - 论文通过一般（各分量独立的）正态分布与标准正态分布的 KL 散度$KL\Big(\mathcal{N}(\mu,\sigma^2)\Big\Vert \mathcal{N}(0,I)\Big)$作为让所有后验看齐标准正态分布的额外损失$\mathcal{L}_{\mu,\sigma^2}=\frac{1}{2} \sum_{i=1}^d \Big(\mu_{(i)}^2 + \sigma_{(i)}^2 - \log \sigma_{(i)}^2 - 1\Big)$
 - 从正态分布$p(z\mid x^{(i)})$中采样$z^{(i)}$的过程不可导，因此通过[Reparameterization Trick](/posts/reparameterization-trick/)将不确定的采样$z^{i} \in \mathcal{N}(\mu^{i}, \sigma^{2})$变为基于随机量$\epsilon \in \mathcal{N}(0, I)$的确定过程$z^{(i)}= \mu^{i} + \epsilon \times \sigma^{i}$ 
VAE 即是一个使用学好的近似推断的有向模型，可以纯粹地使用基于梯度地方法进行训练。
> 编码: $x \to z$，生成: $z \to x$
## 异化的自编码器
VAE 是 AE（自编码器）的一种，但其编码器并非直接编码数据的高维表示，而是基于变分和贝叶斯理论编码数据的均值和方差统计量。
### 编码器
训练过程中 KL loss 希望编码器采样有高斯噪声的潜变量（尽可能接近标准正态分布，因为度量生成数据和真实数据的捷径是方差为零的确定潜变量，此时退化为一般自编码，隐变量没有辨识度，而这是我们不想看到的）；而重构过程中希望不存在噪声（从标准正态采样生成图像）。可以看到 VAE 内部包含了一个对抗的过程。
#### 均值编码器
本质上就是在我们常规的自编码器的基础上，对 encoder 的结果（在 VAE 中对应着计算均值的网络）加上了“高斯噪声”，使得结果 decoder 能够对噪声有鲁棒性；而那个额外的 KL loss（目的是让均值为 0，方差为 1），事实上就是相当于对 encoder 的一个正则项，希望 encoder 出来的东西均有零均值。
#### 方差编码器
用来动态调节噪声的强度，当 decoder 还没有训练好时（重构误差远大于 KL loss），就会适当降低噪声（KL loss 增加），使得拟合起来容易一些（重构误差开始下降）；反之，如果 decoder 训练得还不错时（重构误差小于 KL loss），这时候噪声就会增加（KL loss 减少），使得拟合更加困难了（重构误差又开始增加），这时候 decoder 就要想办法提高它的生成能力了。
### 变分法
对于散度$KL\Big(p(x)\Big\Vert q(x)\Big)$，我们实际上在求一个能使任意概率分布$q(x)$尽可能接近固定概率分布$p(x)$的泛函，其极值$p(x)=q(x)$即是我们求解的目标，可以通过变分法获取其变分下界。
## 单层 VAE
训练时，使用后验 $q \left ( z \mid x \right )$ 获得潜变量 $z$ （编码器）；推断时，从先验 $p_{model} \left ( x \mid z \right )$ 分布中采样 $x$ （解码器）。
![Single VAE](https://s2.loli.net/2023/03/23/FfiumSdzApTwbMt.png)
首先我们可以假设潜变量为标准正态分布，我们希望得到的后验分布应该为：
$$
q(z|x) = \frac{p(x|z)p(z)}{p(x)} = \frac{p(x|z)p(z)}{\int p(x|z)p(z)dz}
$$
其中估计的真实数据分布为：
$$
p_{model} \left ( x \right ) = \int_{z} p \left ( x \mid z \right ) \, p \left ( z \right )
$$
>因为先验分布 $p \left ( x \mid z \right )$ 也无法确定（后验未知和先验未知开始套娃），故该式几乎无法计算，我们通过神经网络分别近似拟合后验（encoder）和先验（decoder）。

对于估计的数据分布，我们可以右边分子分母同时乘以后验（变分）：
$$
p_{model} \left ( x \right ) = \int q \left ( z \mid x \right ) \frac{p_{model} \left ( x \mid z \right ) \, p_{model} \left ( z \right )} {q \left ( z \mid x \right )}
$$
两边同时进行 $\log$，即变为最大似然估计：
$$
\log p_{model} \left ( x \right ) = \log \mathbb{E}_{z \sim q \left ( z \mid x \right )} \left [ \frac{p_{model} \left ( x \mid z \right ) \, p_{model} \left ( z \right )} {q \left ( z \mid x \right )} \right ]
$$
基于 Jensen 不等式可得最大似然的变分下界：
$$
\log p_{model} \left ( x \right ) \ge \mathbb{E}_{z \sim q \left ( z \mid x \right )} \left [ \log \frac{p_{model} \left ( x \mid z \right ) \, p_{model} \left ( z \right )} {q \left ( z \mid x \right )} \right ]
$$
>Jensen 不等式，给出积分的凸函数值和凸函数积分值之间的关系，用概率论描述为 $\varphi \left (  \mathbb{E} \left ( X \right )  \right) \le \mathbb{E} \left ( \varphi \left ( X \right ) \right )$。此处对于 $\log$ 凹函数而言，均值的变换大于等于变换后的均值。

将右项先验分子提出来，潜变量分布和后验分式取倒数：
$$
\log p_{model} \left ( x \right ) \ge \mathbb{E}_{z \sim q \left ( z \mid x \right )} \log p_{model} \left ( x \mid z \right ) - \mathbb{E}_{z \sim q \left ( z \mid x \right )} \log \frac{ q \left ( z \mid x \right ) } { p_{model} \left ( z \right ) }
$$
右边第二项即为后验对潜变量的散度：
$$
\log p_{model} \left ( x \right ) \ge \mathbb{E}_{z \sim q \left ( z \mid x \right )} \log p_{model} \left ( x \mid z \right ) - D_{KL} \left ( q \left ( z \mid x \right ) \parallel p_{model} \left ( z \right ) \right)
$$
右边第一项可以视为潜变量在近似后验下可见和隐藏变量的联合对数似然性，对应解码器生成模型近似；第二项则可视为近似后验的熵，对应编码器后验模型近似。
>对于后验分布采样模型，目标是最小化后验估计分布（假设为正态分布）和潜变量分布（假设为标准正态分布）的 KL 散度。$KL\Big(q(z|x)\Big\Vert p(z)\Big)=\frac{1}{2} \sum_{k=1}^d \Big(\mu_{(k)}^2(x) + \sigma_{(k)}^2(x) - \ln \sigma_{(k)}^2(x) - 1\Big)$

>对于先验生成模型，先从后验模型预测的正太后验分布中采样一个专属于$x$的潜变量$z$（对多个 epoch 而言，每次随机采样一个潜变量可以保证采样的充分性），再选择先验分布，其中伯努利分布（二元分布）对应交叉熵作为损失，而正态分布对应 MSE 损失。

## 多层 VAE
假设多层间的变换符合马尔科夫链过程，即下一层潜变量只取决于上一层的潜变量。

![multiple VAE](https://s2.loli.net/2023/03/23/bdm3ayKZteO27zo.png)

以两层潜变量为例，马尔科夫链过程中存在链式法则：

$$
\begin{array}{c}
p \left ( x , z_{1}, z_{2} \right ) = p \left ( x \mid z_{1} , z_{2} \right ) \, p \left ( z_{1} \mid z_{2} \right ) \, p \left ( z_{2} \right ) \\
q \left ( z_{1} , z_{2} \mid x \right ) = q \left ( z_{1} \mid x \right ) \, q \left ( z_{2} \mid z_{1},x \right )
\end{array}
$$

因为 $z_{2}$ 潜变量与 $x$ 无关：

$$
\begin{array}{c}
p \left ( x , z_{1}, z_{2} \right ) = p \left ( x \mid z_{1} \right ) \, p \left ( z_{1} \mid z_{2} \right ) \, p \left ( z_{2} \right ) \\
q \left ( z_{1} , z_{2} \mid x \right ) = q \left ( z_{1} \mid x \right ) \, q \left ( z_{2} \mid z_{1} \right )
\end{array}
$$

参照单层可推先验的联合概率密度分布为：

$$
p_{model} \left ( x \right ) = \int_{z_1} \int_{z_2} p_{model} \left ( x , z_{1}, z_{2} \right )
$$

同上可得置信下界为：

$$
\begin{aligned}
\log p_{model} \left ( x \right ) &\ge \mathbb{E}_{z_{1}, z_{2} \sim q \left ( z_{1}, z_{2} \mid x \right )} \left [ \log \frac{p_{model} \left ( x, z_{1}, z_{2} \right ) } {q \left ( z_{1}, z_{2} \mid x \right )} \right ]\\
&= \mathbb{E}_{z_{1}, z_{2} \sim q \left ( z_{1}, z_{2} \mid x \right )} \left [ \log \frac{p_{model} \left ( x \mid z_{1} \right ) p_{model} \left ( z_{1} \mid z_{2} \right ) p_{model} \left ( z_{2} \right )} {q \left ( z_{1} \mid x \right ) q \left ( z_{2} \mid z_{1} \right )} \right ]\\
&= \mathbb{E}_{q \left ( z_{1}, z_{2} \mid x \right )} \log p_{model} \left ( x , z_{1}, z_{2} \right ) - D_{KL} \left ( q \left ( z_{1},  z_{2} \mid x \right ) \parallel p_{model} \left ( z_{2} \right ) \right)
\end{aligned}
$$


带入上面推出的条件链式法则，可得损失函数：

$$
\mathcal L = \mathbb{E}_{q \left ( z_{1}, z_{2} \mid x \right )} \left [ \log p \left ( x \mid z_{1} \right ) - \log p \left ( z_{1} \mid x \right ) + \log p \left ( z_{1} \mid z_{2} \right ) - \log q \left ( z_{2} \mid z_{1} \right ) + \log p \left ( z_{2} \right ) \right ]
$$
