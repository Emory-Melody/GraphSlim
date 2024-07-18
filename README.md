# GraphSlim

[//]: # (**[Documentation]&#40;https://graphslim.readthedocs.io/en/latest/&#41;**)
[![Documentation Status](https://readthedocs.org/projects/graphslim/badge/?version=latest)](https://graphslim.readthedocs.io/en/latest/?badge=latest)

**[Benchmark Paper](https://arxiv.org/abs/2406.16715)** |
**[Benchmark Scripts](https://github.com/Emory-Melody/GraphSlim/tree/main/benchmark)** |
**[Survey Paper](https://arxiv.org/pdf/2402.03358)** |
**[Paper Collection](https://github.com/Emory-Melody/awesome-graph-reduction)**

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

Please choose from `requirements_torch1+.txt` and `requirements_torch2+.txt` at your convenience.
Please change the cuda version of `torch`, `torch-geometric` and `torch-sparse` in the requirements file according to
your system configuration.

<!--# Download Datasets

For cora, citeseer, flickr and reddit (reddit2 in pyg), the pyg code will directly download them.
For arxiv, we use the datasets provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT). 
Our code will automatically download all datasets.

The default path of datasets is `../../data`.-->


[//]: # (# Installation)

[//]: # ()

[//]: # (## Install from pip)

[//]: # ()

[//]: # (```)

[//]: # (pip install graphslim)

[//]: # (```)

# Examples

```
python examples/train_coreset.py
python examples/train_coarsen.py
python examples/train_gcond.py
```

See more examples in **[Benchmark Scripts](https://github.com/Emory-Melody/GraphSlim/tree/main/benchmark)**.

# Usage

## Command Line

Run `python configs.py --help` and you will see

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

## Package Style

```python
from graphslim.configs import cli
from graphslim.dataset import *
from graphslim.evaluation import *
from graphslim.sparsification import Random

args = cli(standalone_mode=False)
graph = get_dataset(args.dataset, args)
# To reproduce the benchmark, use our args and graph class
# To use your own args and graph format, please ensure the args and graph class has the required attributes

# create an agent of one reduction algorithm
agent = Random(setting=args.setting, data=graph, args=args)
# reduce the graph 
reduced_graph = agent.reduce(graph, verbose=args.verbose)
# create an evaluator
evaluator = Evaluator(args)
# evaluate the reduced graph on a GNN model
res_mean, res_std = evaluator.evaluate(reduced_graph, model_type='GCN')
```

# Customization

* To implement a new reduction algorithm, you need to create a new class in `sparsification` or `coarsening`
  or `condensation` and inherit the `Base` class.
* To implement a new dataset, you need to create a new class in `dataset/loader.py` and inherit the `TransAndInd` class.
* To implement a new evaluation metric, you need to create a new function in `evaluation/eval_agent.py`.
* To implement a new GNN model, you need to create a new class in `models` and inherit the `Base` class.
* To customize sparsification before evaluation, please modify the function `sparsify` in `evaluation/utils.py`.

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

[GCOND](https://github.com/ChandlerBang/GCond)

[GCSNTK](https://github.com/WANGLin0126/GCSNTK)

[SFGC](https://github.com/Amanda-Zheng/SFGC)

[GEOM](https://github.com/NUS-HPC-AI-Lab/GEOM/tree/main)

[DeepRobust](https://github.com/DSE-MSU/DeepRobust)

