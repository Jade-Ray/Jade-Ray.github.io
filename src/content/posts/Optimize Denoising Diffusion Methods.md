---
title: Optimize Denoising Diffusion Methods
date: 2024-02-01T08:15:20.424Z
tags: []
draft: false
---
# Loss

## Min-SNR weighting Stragegy

[Min-SNR](https://arxiv.org/abs/2303.09556)方法认为时间步长间的优化方向存在冲突是去噪扩散模型收敛速度慢的部分原因。因此该方法将扩散训练视为多任务学习问题（每个时间步长内的去噪过程视为单独任务），根据钳位信噪比（`SNR`）调整时间步长的损失权重，有效平衡时间步长之间的冲突。

>However, it’s important to keep in mind that different steps may have vastly different requirements. At each step of a diffusion model, the strength of the denoising varies. For example, easier denoising tasks (when t → 0) may require simple reconstructions of the input in order to achieve lower denoising loss. This strategy, unfortunately, does not work as well for noisier tasks (when t → T ). Thus, it’s extremely important to analyze the correlation between different timesteps.

包含每个时间步长下去噪损失的多任务损失函数中，关于模型参数$\theta$的收敛方向$\delta^{\ast}$表示为：

$$
\delta^{\ast} = - \sum_{t=1}^{T} w_{t}\nabla_{\theta}\mathcal{L}^{t}(\theta)
$$

其中$w_{t}$表示为不同时间步长对应的权重，因为模型预测目标为噪声，传统多任务优化方法获得权重的方式低效且不稳定。因此该方法通过信噪比加权方式优化，对于任意步长加噪结果$x_{t}= \alpha_{t}x_{0} + \sigma_{t}\epsilon$，其加权信噪比为$w_{t}=SNR(t)=\alpha_{t}^{2} / \sigma_{t}^{2}$，对于噪声目标它在数值上等价于常加权策略。进一步为了避免模型过多关注小噪声水平，提出$\text{Min-SNR-}\gamma$ 加权策略$w_{t} = min \left\{ SNR(t), \gamma \right\}$。
