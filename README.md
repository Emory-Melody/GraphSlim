# GraphSlim

[//]: # (**[Documentation]&#40;https://graphslim.readthedocs.io/en/latest/&#41;**)

**[Benchmark Paper]()** | **[Benchmark Scripts](https://github.com/rockcor/graphslim/tree/master/benchmark)** | 
**[Survey Paper](https://arxiv.org/pdf/2402.03358)** | **[Paper Collection](https://github.com/Emory-Melody/awesome-graph-reduction)**

# Features

GraphSlim is a PyTorch library for graph reduction. It takes graph of PyG format as input and outputs a reduced graph preserving **properties or performance** of the original graph.

* Covering representative methods of all 3 graph reduction strategies: Sparsification, Coarsening and Condensation.
* Different reduction strategies can be easily combined in one run.
* Unified evaluation tools including Grid Search and NAS.
* Support evasion and poisoning attacks on the input graph by DeepRobust.

# Guidance

* Please first prepare the environment and datasets.
* If you are new to GraphSlim, we highly suggest you first run the examples in the `examples` folder.
* If you have any questions or suggestions regarding this library, feel free to create an
  issue [here](https://github.com/rockcor/graphslim/issues). We will reply as soon as possible :)

# Prepare Environments

Please choose from `requirements_torch1+.txt` and `requirements_torch2+.txt` at your convenience. 
Please change the cuda version of `torch`, `torch-geometric` and `torch-sparse` in the requirements file according to
your system configuration.

# Download Datasets

For cora, citeseer, flickr and reddit (reddit2 in pyg), the pyg code will directly download them.
For arxiv, we use the datasets provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT). Our code will automatically download it.

The default path of datasets is `../../data`.


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

See more examples in **[Benchmark Scripts](https://github.com/rockcor/graphslim/tree/master/benchmark)**.
# Usage

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

# Limitations

* The GEOM and SFGC are not fully implemented in the current version due to disk space limit. We set the number of
  experts to 20 currently.

# Acknowledgement

Some of the algorithms are referred to paper authors' implementations and other packages.

[SCAL](https://github.com/szzhang17/Scaling-Up-Graph-Neural-Networks-Via-Graph-Coarsening)

[GCOND](https://github.com/ChandlerBang/GCond)

[GCSNTK](https://github.com/WANGLin0126/GCSNTK)

[SFGC](https://github.com/Amanda-Zheng/SFGC)

[GEOM](https://github.com/NUS-HPC-AI-Lab/GEOM/tree/main)

[DeepRobust](https://github.com/DSE-MSU/DeepRobust)

