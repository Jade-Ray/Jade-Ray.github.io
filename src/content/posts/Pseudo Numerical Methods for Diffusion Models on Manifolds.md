---
tags:
  - Diffusion
  - SDE
aliases:
  - PNDM
title: Pseudo Numerical Methods for Diffusion Models on Manifolds
date: 2024-02-01T11:29:16.235Z
draft: false
---
在[Denoising Diffusion Implict Models](/posts/denoising-diffusion-implict-models/#从数值方法理解ddim-常微分方程-ode)中，我们可以进一步寻找其对应的常微分方程为：

$$
x_{t-\delta} - x_{t} = \left( \bar{\alpha}_{t-\delta} - \bar{\alpha}_{t} \right) \left( \frac{x_{t}}{\sqrt{\bar{\alpha}_{t}} \left(\sqrt{\bar{\alpha}_{t-\delta}} + \sqrt{\bar{\alpha}_{t}} \right)} - \frac{\epsilon_{\theta} \left( x_{t},\, t \right)}{\sqrt{\bar{\alpha}_{t}} \left(\sqrt{(1 - \bar{\alpha}_{t-\delta})\bar{\alpha}_{t}} + \sqrt{(1 - \bar{\alpha}_{t})\bar{\alpha}_{t-\delta}} \right)} \right)
$$
