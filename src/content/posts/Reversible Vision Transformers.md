---
tags:
  - Transformers
  - ViT
aliases: RevViT
title: Reversible Vision Transformers
date: 2026-01-16T05:38:25.729Z
draft: false
---
 By decoupling the GPU memory requirement from the depth of the model, Reversible Vision Transformers enable scaling up architectures with efficient memory usage.

- Condider two transformation $T_{1}$ and $T_{2}$ , the input $I$ partitioned into two $d$ dimensional tensors $\left [  I_{1};I_{2} \right ]$ , output tensor $O$ also similarly partitioned into $\left [ O_{1};O_{2} \right ]$ . $F$ and $G$ are arbitrary differentiable function.
$$
\begin{array}{l}
I = \begin{bmatrix} I_{1}\\ I_{2}\end{bmatrix} \underset{T_{1}}{\longrightarrow} \begin{bmatrix} O_{1}\\ O_{2}\end{bmatrix} = \begin{bmatrix} I_{1}\\ I_{2} + F\left ( I_{1} \right ) \end{bmatrix} = O \\
I = \begin{bmatrix} I_{1}\\ I_{2}\end{bmatrix} \underset{T_{2}}{\longrightarrow} \begin{bmatrix} O_{1}\\ O_{2}\end{bmatrix} = \begin{bmatrix} I_{1} + G\left ( I_{1} \right )\\ I_{2} \end{bmatrix} = O
\end{array}
$$
above transformation $T_{1}$ and $T_{2}$ allow an inverse transformation ${T_{1}}'$ and ${T_{2}}'$ . Now, consider compositon $T = T_{2} \circ T_{1}$ :

$$
I = \begin{bmatrix} I_{1}\\ I_{2}\end{bmatrix} \underset{T}{\longrightarrow} \begin{bmatrix} O_{1}\\ O_{2}\end{bmatrix} = \begin{bmatrix} I_{1} + G\left ( I_{2} + F\left ( I_{1} \right ) \right ) \\ I_{2} + F\left ( I_{1} \right ) \end{bmatrix} = O
$$

Naturally, ${T}' = {T_{2}}' \circ {T_{1}}'$ that follows ${T}' \left( T \left( I \right) \right) = I$ .

---
### Construction
- Reversible ViT: a two-residual-stream architecture composed of a stack of Reversible ViT blocks (a)
- Reversible MViT: a two-residual-stream architecture as well
 - Stage-Transition: coupling between the residual streams as well as perform channel upsampling and resolution downsampling.
 - Stage-Preserving: form the majority of the computational graph and propagate information preserving input featuren dimension.

![Block Struct](https://d3i71xaburhd42.cloudfront.net/665348fc446dd5185c93a5be4c766dad43186e6b/4-Figure2-1.png)
