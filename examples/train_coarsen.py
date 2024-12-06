import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from graphslim.config import get_args
from graphslim.dataset import *
from graphslim.evaluation import *
from graphslim.coarsening import *
from graphslim.utils import to_camel_case, seed_everything

args = get_args()
graph = get_dataset(args.dataset, args, args.load_path)
seed_everything(args.seed)
agent = VariationNeighborhoods(setting=args.setting, data=graph, args=args)
reduced_graph = agent.reduce(graph)
evaluator = Evaluator(args)
evaluator.evaluate(reduced_graph, 'GCN')
