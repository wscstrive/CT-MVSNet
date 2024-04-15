<h1 align="center">CT-MVSNet: Efficient Multi-View Stereo with Cross-scale Transformer(MMM Oral 2024)</h1>

<div align="center">
    <a href="https://github.com/wscstrive" target='_blank'>Sicheng Wang</a>, 
    <a href="https://faculty.nuist.edu.cn/jianghao/zh_CN/index.htm" target='_blank'>Hao Jiang</a>*, 
    <a href="https://github.com/Sion1" target='_blank'>Lei Xiang</a>, 
</div>

<br />

<div align="center">
    <a href="https://link.springer.com/chapter/10.1007/978-3-031-53308-2_29" target='_blank'><img src="https://img.shields.io/badge/MMM-2024-1E90FF"/></a>&nbsp;
    <a href="https://arxiv.org/abs/2312.08594.pdf" target='_blank'><img src="https://img.shields.io/badge/arxiv-arxiv.2312.08594-B31B1B?logo=arXiv&logoColor=green"/></a>&nbsp;
</div>

<br />

<div align="center">

## 📔 Introduction
In this paper, we propose a novel cross-scale transformer (CT) that processes feature representations at different stages without additional computation. Specifically, we introduce an adaptive matching-aware transformer (AMT) that employs different interactive attention combinations at multiple scales. This combined strategy enables our network to capture intra-image context information and enhance inter-image feature relationships. Besides, we present a dual-feature guided aggregation (DFGA) that embeds the coarse global semantic information into the finer cost volume construction to further strengthen global and local feature awareness. Meanwhile, we design a feature metric loss (FM Loss) that evaluates the feature bias before and after transformation to reduce the impact of feature mismatch on depth estimation. Extensive experiments on DTU dataset and Tanks and Temples benchmark demonstrate that our method achieves state-of-the-art results.
![](asserts/overview.png)
