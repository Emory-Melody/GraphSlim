# Preparation
## Requirements

Please see `requirements.txt`.

## Download Datasets

<!--For cora, citeseer and pubmed, the code will directly download them; so no extra script is needed.
For reddit, flickr and arxiv, we use the datasets provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT).
They are available on [Google Drive link](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (alternatively, [BaiduYun link (code: f1ao)](https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg)). 
Rename the folder to `data` at the root directory. Note that the links are provided by GraphSAINT team.-->

Put all datasets in `../../data` in your system!

# Abstract

Graph reduction for all graph algorithms especially for graph neural networks (GNNs).
This package aims to reduce the large, original graph into a small, synthetic and highly-informative graph.

# Features
* Covering 3 mainstream reduction strategies: Sparsificaiton, Coarsening and Condensation
* Unified test tools for easily producing benchmarks
