---
tags:
  - VLM
aliases:
  - CogVLM
title: Visual Expert for Pretrained Language Models
date: 2025-02-14T11:02:53.548Z
draft: false
---
其他视觉语言基础大模型面对视觉-文本多模态输入时，主流方法是将图像特征映射到文本特征空间中。CogVLM 方法另辟蹊径，通过可训练的视觉专家模块分别在解码器的注意力层和 FFN 层融合并对齐视觉特征和文本特征。
![image.png](https://s2.loli.net/2025/02/14/tphjOnVMKBer8YN.png)
## Visuall expert module
语言模型的每个注意力头都捕获了语义信息的某些方面，而可学习的视觉专家模块可以将图像特征转化为视觉注意力头并对齐到不同的语言注意力头中。
假设注意力层的输入为 $X \in \mathbb{R}^{B\times H\times (L_{I}+L_{T})\times D}$，其中 $L_{I},L_{T}$ 分别是图像和文本的长度， $H$ 为注意力头数， $B$ 为批数， $D$ 为潜变量维数。在视觉专家注意力中，首先将 $X$ 分割为图像潜变量 $X_{I}$ 和文本潜变量 $X_{T}$，注意力计算过程如下：

$$
\begin{aligned}
\text{Attention}(X&,W_{I},W_{T}) = \text{softmax} \left(\frac{\text{Tril}(QK^{T})}{\sqrt{D}} \right)V,\\
Q &= \text{concat}(X_{I}W_{I}^{Q},X_{T}W_{T}^{Q}), \\
K &= \text{concat}(X_{I}W_{I}^{K},X_{T}W_{T}^{K}), \\
V &= \text{concat}(X_{I}W_{I}^{V},X_{T}W_{T}^{V}),
\end{aligned}
$$

其中 $\text{Tril}(\cdot)$ 表示下三角遮罩。此外在 FFN 层有类似改动：

$$
\text{FFN}(X) = \text{concat}(\text{FFN}_{I}(X_{I}),\text{FFN}_{T}(X_{T}))
$$

>在文本中使用下三角遮罩维持其自回归性，屏蔽未来位置的信息，保证生成的因果性是常见的。但将其推广到图像序列上有点奇怪，虽然 ViT 是将图像分块后提取的视觉信息，但图像块序列本身并没有严格的时序约束，所有图像块才能包含整体图像信息。这里只能看作是为了确保文本生成的自回归一致性，强制在铺平的视觉序列上遵循推理的因果约束，控制跨模态信息流的方向。

## 训练流程
同样分为三步流程，前两步为预训练，后一步为多任务微调。
### pre-training
- 阶段一：训练图像描述任务，在 1.5B LAION-2B 和 COYO-700M 数据集上训练。
- 阶段二：混合图像描述任务和 Referring Expression Comprehension (REC) 任务，预测目标文本描述的边界框，类似 VQA 形式 `Question: Where is the Object?` `Answer: [x0,y0,x1,y1](/posts/x0-y0-x1-y1/)`。
![image.png](https://s2.loli.net/2025/02/14/dVtmUE6wS4i5eCu.png)
### multitask finetuning
针对具体任务（如视觉问答、图像描述、视觉推理）优化模型的多模态交互能力。共训练两种通用模型：
- CogVLM-Chat：接受自然语言输入和输出，使用 VQA v2、TextCaps、ScienceQA 等混合多任务数据集数据
- CogVLM-Grounding：接受带目标框的自然语言输入和输出，包括 Flickr30K、Ref-COCO、Visual7W、VisualGenome 和 Grounded CoT-VQA 数据集，涵盖四类高质量 grounding 数据集：
 - Grounded Captioning (GC)：图像描述中的每个名词短语之后是相应的目标框；
 - Referring Expression Generation (REG)：图像中的每个目标框被描述性文本准确表达对应内容在图像中的区域；
 - Referring Expression Comprehension (REC)：每个文本描述包括多个关联目标的对应框；
 - Grounded Visual Question Answering (Grounded VQA)：VQA 风格的可能包含参考区域的提问。
![image.png](https://s2.loli.net/2025/02/14/5RM8wiYkOKj2gbN.png)
## 后续提升
在 CogAgent 中提出双分辨率编码器（低分辨率全局编码器 + 高分辨率局部编码器）在少量提升算力的前提下将分辨率从最高 $490\times 490$提升至 $1120\times 1120$。
![image.png](https://s2.loli.net/2025/02/14/qfBRlLb7UjXiFDM.png)
对于带低分辨率的解码器输入 $X_{in} \in \mathbb{R}^{B\times (L_{I_{lo}}+L_{T})\times D_{dec}}$，带高分辨的图像编码器输出 $X_{hi} \in \mathbb{R}^{B\times L_{I_{hi}}} \times D_{hi}$，每层通过交叉注意力模块实现高分辨和低分辨率的融合：

$$
\begin{aligned}
X_{i} &= \text{MSA}(\text{layernorm}(X_{in})) + X_{in} \\
X_{out} &= \text{MCA}(\text{layernorm}(X_{i}),X_{hi}) + X_{i}
\end{aligned}
$$

时间复杂度从 $O \left( (L_{I_{hi}}+L_{T})^{2} \right)$ 到 $O \left( (L_{I_{lo}}+L_{T})^{2} + (L_{I_{lo}}+L_{T})L_{I_{hi}} \right)$，若 $L_{I_{lo}} \ll L_{I_{hi}}$，则时间复杂度减为 $O \left( L_{T}^{2} + L_{T}L_{I_{hi}} \right)$。
此外，新增 GUI Agent 能力，能够根据用户指令生成具体的操作步骤（如点击按钮、输入文本）并输出操作坐标，通过**动作历史记录**和**结构化输出**（自然语言描述 + 函数调用式参数）实现多步任务规划。
