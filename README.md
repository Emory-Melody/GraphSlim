<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> (NeurIPS'25) GraphSlim, a PyTorch Library for Graph Reduction. </b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/Emory-Melody/GraphSlim?color=green)
![](https://img.shields.io/github/stars/Emory-Melody/GraphSlim?color=yellow)
![](https://img.shields.io/github/forks/Emory-Melody/GraphSlim?color=lightblue)
![](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)
![](https://img.shields.io/badge/PRs-Welcome-green)

</div>

<div align="center">

**[<a href="https://graphslim.readthedocs.io/en/latest/?badge=latest">Documentation</a>]**
**[<a href="https://graphslim-vis.streamlit.app/">Web Interface</a>]**
**[<a href="https://colab.research.google.com/drive/1LLG9PYOPnmLCAr0ow0ogRYI8DvLGPk7d?usp=sharing">Online Demo</a>]**

**[<a href="https://arxiv.org/abs/2406.16715">NeurIPS'25 Benchmark Paper</a>]**
**[<a href="https://github.com/Emory-Melody/GraphSlim/tree/main/benchmark">Benchmark Scripts</a>]**
**[<a href="https://arxiv.org/pdf/2402.03358">IJCAI'24 Survey Paper</a>]**
**[<a href="https://github.com/Emory-Melody/awesome-graph-reduction">Paper Collection</a>]**

</div>

<div align="center">
  <img src="./figures/logo.png" width="300">
</div>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```
@inproceedings{
gong2025gcnc,
title={{GC}4{NC}: A Benchmark Framework for Graph Condensation on Node Classification with New Insights},
author={Shengbo Gong and Juntong Ni and Noveen Sachdeva and Carl Yang and Wei Jin},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2025},
url={https://openreview.net/forum?id=ZhxeUImT89}
}
```

# Features

GraphSlim is a PyTorch library for graph reduction. It takes graph of PyG format as input and outputs a reduced graph
preserving **properties or performance** of the original graph.

* Covering representative methods of all 3 graph reduction strategies: Sparsification, Coarsening and Condensation.
* Different reduction strategies can be easily combined in one run.
* Unified evaluation tools including Grid Search and NAS.
* Support evasion and poisoning attacks on the input graph by DeepRobust.

# Guidance

* Please first prepare the environments.
* If you are new to GraphSlim, we highly suggest you first run the examples in the `examples` folder.
* If you have any questions or suggestions regarding this library, feel free to create an
  issue [here](https://github.com/Emory-Melody/GraphSlim/issues). We will reply as soon as possible :)

# Prepare Environments

## CUDA and PyTorch
Check [torch previous versions](https://pytorch.org/get-started/previous-versions/).
We test  this repo  in  `torch 1.13.1`  and `torch  2.1.2` with `CUDA 12.4`.

## Install from requirements

Please choose from `requirements_torch1+.txt (for torch 1.\*)` and `requirements.txt (for torch2.*)` at your convenience.

<!--# Download Datasets

For cora, citeseer, flickr and reddit (reddit2 in pyg), the pyg code will directly download them.
For arxiv, we use the datasets provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT). 
Our code will automatically download all datasets.

The default path of datasets is `../../data`.-->

## Install from pip

```shell
# choose one version from https://data.pyg.org/whl/ based on your environment
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install graphslim
```

# Examples

```
python examples/train_coreset.py
python examples/train_coarsen.py
python examples/train_gcond.py
```

See more examples in **[Benchmark Scripts](https://github.com/Emory-Melody/GraphSlim/tree/main/benchmark)**.

# Use As Project

```shell
cd graphslim
python train_all.py -xxx xx
```

Run `python configs.py --help` to get all command line options.

```shell
Options:
  -D, --dataset TEXT              [default: cora]
  -G, --gpu_id INTEGER            gpu id start from 0, -1 means cpu  [default:
                                  0]
  --setting [trans|ind]           transductive or inductive setting
  --split TEXT                    only support public split now, do not change
                                  it  [default: fixed]
  --run_reduction INTEGER         repeat times of reduction  [default: 3]
  --run_eval INTEGER              repeat times of final evaluations  [default:
                                  10]
  --run_inter_eval INTEGER        repeat times of intermediate evaluations
                                  [default: 5]
  --eval_interval INTEGER         [default: 100]
  -H, --hidden INTEGER            [default: 256]
  --eval_epochs, --ee INTEGER     [default: 300]
  --eval_model, --em [GCN|GAT|SGC|APPNP|Cheby|GraphSage|GAT|SGFormer]
                                  [default: GCN]
  --condense_model [GCN|GAT|SGC|APPNP|Cheby|GraphSage|GAT]
                                  [default: SGC]
  -E, --epochs INTEGER            number of reduction epochs  [default: 1000]
  --lr FLOAT                      [default: 0.01]
  --weight_decay, --wd INTEGER    [default: 0]
  --pre_norm BOOLEAN              pre-normalize features, forced true for
                                  arxiv, flickr and reddit  [default: True]
  --outer_loop INTEGER            [default: 10]
  --inner_loop INTEGER            [default: 1]
  -R, --reduction_rate FLOAT      -1 means use representative reduction rate;
                                  reduction rate of training set, defined as
                                  (number of nodes in small graph)/(number of
                                  nodes in original graph)  [default: -1.0]
  -S, --seed INTEGER              Random seed  [default: 1]
  --nlayers INTEGER               number of GNN layers of condensed model
                                  [default: 2]
  -V, --verbose
  --init [variation_neighborhoods|variation_edges|variation_cliques|heavy_edge|algebraic_JC|affinity_GS|kron|vng|clustering|averaging|cent_d|cent_p|kcenter|herding|random]
                                  features initialization methods
  -M, --method [variation_neighborhoods|variation_edges|variation_cliques|heavy_edge|algebraic_JC|affinity_GS|kron|vng|clustering|averaging|gcond|doscond|gcondx|doscondx|sfgc|msgc|disco|sgdd|gcsntk|geom|cent_d|cent_p|kcenter|herding|random]
                                  [default: kcenter]
  --activation [sigmoid|tanh|relu|linear|softplus|leakyrelu|relu6|elu]
                                  activation function when do NAS  [default:
                                  relu]
  -A, --attack [random_adj|metattack|random_feat]
                                  corruption method
  -P, --ptb_r FLOAT               perturbation rate for corruptions  [default:
                                  0.25]
  --aggpreprocess                 use aggregation for coreset methods
  --dis_metric TEXT               distance metric for all condensation
                                  methods,ours means metric used in GCond
                                  paper  [default: ours]
  --lr_adj FLOAT                  [default: 0.0001]
  --lr_feat FLOAT                 [default: 0.0001]
  --threshold INTEGER             sparsificaiton threshold before evaluation
                                  [default: 0]
  --dropout FLOAT                 [default: 0.0]
  --ntrans INTEGER                number of transformations in SGC and APPNP
                                  [default: 1]
  --with_bn
  --no_buff                       skip the buffer generation and use existing
                                  in geom,sfgc
  --batch_adj INTEGER             batch size for msgc  [default: 1]
  --alpha FLOAT                   for appnp  [default: 0.1]
  --mx_size INTEGER               for gcsntk methods, avoid SVD error
                                  [default: 100]
  --save_path, --sp TEXT          save path for synthetic graph  [default:
                                  ../checkpoints]
  -W, --eval_whole                if run on whole graph
  --help                          Show this message and exit.
```

# Use As Package

```python
from graphslim.dataset import *
from graphslim.evaluation import *
from graphslim.condensation import GCond
from graphslim.config import cli

args = cli(standalone_mode=False)
# customize args here
args.reduction_rate = 0.5
args.device = 'cuda:0'
# add more args.<main_args/dataset_args> here
graph = get_dataset('cora', args=args)
# To reproduce the benchmark, use our args and graph class
# To use your own args and graph format, please ensure the args and graph class has the required attributes
# create an agent of one reduction algorithm
# add more args.<agent_args> here
agent = GCond(setting='trans', data=graph, args=args)
# reduce the graph 
reduced_graph = agent.reduce(graph, verbose=True)
# create an evaluator
# add more args.<evaluator_args> here
evaluator = Evaluator(args)
# evaluate the reduced graph on a GNN model
res_mean, res_std = evaluator.evaluate(reduced_graph, model_type='GCN')
```

All parameters can be divided into

```shell
<main_args>: dataset, method, setting, reduction_rate, seed, aggpreprocess, eval_whole, run_reduction
<attack_args>: attack, ptb_r
<dataset_args>: pre_norm, save_path, split, threshold
<agent_args>: init, eval_interval, eval_epochs, eval_model, condense_model, epochs, lr, weight_decay, outer_loop, inner_loop, nlayers, method, activation, dropout, ntrans, with_bn, no_buff, batch_adj, alpha, mx_size, dis_metric, lr_adj, lr_feat
<evaluator_args>: final_eval_model, eval_epochs, lr, weight_decay
```

See more details
in [![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://graphslim.readthedocs.io/en/latest/?badge=latest)

# Customization

* To implement a new reduction algorithm, you need to create a new class in `sparsification` or `coarsening`
  or `condensation` and inherit the `Base` class.
* To implement a new dataset, you need to create a new class in `dataset/loader.py` and inherit the `TransAndInd` class.
* To implement a new evaluation metric, you need to create a new function in `evaluation/eval_agent.py`.
* To implement a new GNN model, you need to create a new class in `models` and inherit the `Base` class.
* To customize sparsification before evaluation, please modify the function `sparsify` in `evaluation/utils.py`.

# Web Interface

Our [web application](https://graphslim-vis.streamlit.app/) is deployed online using [streamlit](https://streamlit.io/).
But it also can be initiated using:

```bash
cd interface
python -m streamlit run vis_graphslim.py
```

to activate the interface. Please satisfy the dependency in [interface/requirements.txt](interface/requirements.txt).

# TODO

- [ ] Add sparsification algorithms like Spanner
- [ ] Add latest condensation methods
- [ ] Support more datasets
- [ ] Present full results in a website

# Limitations

* The GEOM and SFGC are not fully implemented in the current version due to disk space limit. We set the number of
  experts to 20 currently. If you have over 100GB disk space, you can set the number of experts to 1000 to reproduce the
  If you have over 100GB disk space, you can set the number of experts to 200 to reproduce the results in the paper.

# Acknowledgement

Some of the algorithms are referred to paper authors' implementations and other packages.

[SCAL](https://github.com/szzhang17/Scaling-Up-Graph-Neural-Networks-Via-Graph-Coarsening)

[Sparsification](https://github.com/yuhanchan/sparsification)

[GCOND](https://github.com/ChandlerBang/GCond)

[GCSNTK](https://github.com/WANGLin0126/GCSNTK)

[SFGC](https://github.com/Amanda-Zheng/SFGC)

[GEOM](https://github.com/NUS-HPC-AI-Lab/GEOM/tree/main)

[DeepRobust](https://github.com/DSE-MSU/DeepRobust)

