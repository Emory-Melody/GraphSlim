# GraphSlim

**[Documentation](https://graphslim.readthedocs.io/en/latest/)**

**[Benchmark Paper]()** | **[Benchmark Scripts](https://github.com/rockcor/graphslim/tree/master/benchmark)**

Graph Slim is a PyTorch library for graph reduction methods.

* If you are new to DeepRobust, we highly suggest you read
  the [documentation page](https://graphslim.readthedocs.io/en/latest/) or the following content in this README to learn
  how to use it.
* If you have any questions or suggestions regarding this library, feel free to create an
  issue [here](https://github.com/rockcor/graphslim/issues). We will reply as soon as possible :)

# Basic Environment

see `requirements.txt` for more information.

# Installation

## Install from pip

```
pip install graphslim
```

# Test Examples

```
python examples/train_all.py -D cora -R 0.5 -M random
```

# Usage

```python
from graphslim.configs import cli
from graphslim.dataset import *
from graphslim.evaluation import *

args = cli(standalone_mode=False)
graph = get_dataset(args.dataset, args)

from graphslim.sparsification import Random

agent = Random(setting=args.setting, data=graph, args=args)

reduced_graph = agent.reduce(graph, verbose=args.verbose)
evaluator = Evaluator(args)
res_mean, res_std = evaluator.evaluate(reduced_graph, model_type='GCN')
```

## Acknowledgement

Some of the algorithms are referred to paper authors' implementations and other packages.

[DeepRobust](https://github.com/DSE-MSU/DeepRobust)
