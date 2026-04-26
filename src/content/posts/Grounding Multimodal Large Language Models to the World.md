---
tags:
  - VLM
aliases:
  - KOSMOS-2
title: Grounding Multimodal Large Language Models to the World
date: 2025-02-02T07:54:29.504Z
draft: false
---
Multimodal Large Language Model (MLLM)，实现目标感知描述（如边界框）和通用的文本-视觉关联感知。对于如何构建文本-视觉通用数据集，KOSMOS-2 期望通过类似 Markdown 中超文本标记的形式链接文本和对应描述的图像目标，即 `[text span](bounding boxes)`，将文本描述对象的图像位置转为文本 tokens 再加入到文本中。基于此理念构建的数据集 [GRIT](https://huggingface.co/datasets/zzliang/GRIT)，其数据构成大致为：
```json
{
	# other params omit
	'caption': 'a wire hanger with a paper cover that reads we heart our customers', 
	'width': 1024, 
	'height': 693, 
	'noun_chunks': [[19, 32, 0.019644069503434333, 0.31054004033406574, 0.9622142865754519, 0.9603442351023356, 0.79298526], [0, 13, 0.019422357885505368, 0.027634161214033764, 0.9593302408854166, 0.969467560450236, 0.67520964]], 
	'ref_exps': [[19, 66, 0.019644069503434333, 0.31054004033406574, 0.9622142865754519, 0.9603442351023356, 0.79298526], [0, 66, 0.019422357885505368, 0.027634161214033764, 0.9593302408854166, 0.969467560450236, 0.67520964]],
}
```
其中 `noun_chucks` 表示具有关联边界框的名词块列表，每个名词块包含 `开始位置`、`结束位置`、`x_min`、`y_min`、`x_max`、`y_max`、`置信分数`；此外 `ref_exps` 表示名词块的参考扩展，若无扩展即为本身的复制，具体构建管道如下图所示：
![image.png](https://s2.loli.net/2025/02/02/ynMT5g4qhZ6BSLC.png)
值得注意的是，对于所有扩展的名词块，只保留不包含其他的块，避免名词块的复杂归属关系扰乱文本主语指代的目标。
## 位置 Tokens
显而易见，目标边界框位置由连续的数学坐标表示，这并不利于网络理解视觉信息，也不利于文本的统一建模，将连续坐标转化为一系列离散位置 tokens 是合理的选择，对于 $W \times H$ 图像，均分为 $P$ 段，得到 $P \times P$ 个 $\frac{W}{P}\times \frac{H}{P}$ 像素块，每个块为一个位置 token，表示该块内的所有坐标。
```html
<s> <image> Image Embedding </image> <grounding> <p> It </p><box><loc44><loc863></box> seats next to <p> a campfire </p><box><loc4><loc1007></box> </s>
```
具体文本-图像对输入如上面所示，其中 `<s> </s>` 表示序列范围； `<image> </image>` 表示图像特征嵌入； `<grounding>` 表示后续文本将关联到视觉图像中； `<p> </p>` 表示文本名词块范围； `<box> </box>` 表示对应的目标位置，其中前一个位置 token 为左上角点，后一个为右下角点。
## 结果
![image.png](https://s2.loli.net/2025/02/02/RUzgYSIl49WhbmT.png)
多目标检测
![image.png](https://s2.loli.net/2025/02/02/Unf1oVswJDvdzRQ.png)
图像描述
