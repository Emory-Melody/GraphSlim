import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from graphslim.config import get_args
from graphslim.dataset import *
from graphslim.evaluation import *
from graphslim.sparsification import *
from graphslim.condensation import *
from graphslim.coarsening import *
from graphslim.utils import to_camel_case, seed_everything

if __name__ == '__main__':
    args = get_args()
    graph = get_dataset(args.dataset, args, args.load_path)
    seed_everything(args.seed)
    if args.attack is not None:
        data = attack(graph, args)
    if args.method == 'kcenter' and not args.agg:
        agent = KCenter(setting=args.setting, reduction_rate=args.reduction_rate, data=graph, args=args)
    elif args.method == 'kcenter' and args.agg:
        agent = KCenterAgg(setting=args.setting, data=graph, args=args)
    elif args.method == 'herding' and not args.agg:
        agent = Herding(setting=args.setting, data=graph, args=args)
    elif args.method == 'herding' and args.agg:
        agent = HerdingAgg(setting=args.setting, data=graph, args=args)
    elif args.method == 'random':
        agent = Random(setting=args.setting, data=graph, args=args)
    elif args.method == 'cent_p':
        agent = CentP(setting=args.setting, data=graph, args=args)
    elif args.method == 'cent_d':
        agent = CentD(setting=args.setting, data=graph, args=args)
    elif args.method == 'simgc':
        agent = SimGC(setting=args.setting, data=graph, args=args)
    elif args.method == 'gcond':
        agent = GCond(setting=args.setting, data=graph, args=args)
    elif args.method == 'doscond':
        agent = DosCond(setting=args.setting, data=graph, args=args)
    elif args.method in ['doscondx']:
        agent = DosCondX(setting=args.setting, data=graph, args=args)
    elif args.method in ['gcondx']:
        agent = GCondX(setting=args.setting, data=graph, args=args)
    elif args.method == 'sfgc':
        agent = SFGC(setting=args.setting, data=graph, args=args)
    elif args.method == 'sgdd':
        agent = SGDD(setting=args.setting, data=graph, args=args)
    elif args.method == 'gcsntk':
        agent = GCSNTK(setting=args.setting, data=graph, args=args)
    elif args.method == 'msgc':
        agent = MSGC(setting=args.setting, data=graph, args=args)
    elif args.method == 'geom':
        agent = GEOM(setting=args.setting, data=graph, args=args)
    elif args.method == 'vng':
        agent = VNG(setting=args.setting, data=graph, args=args)
    elif args.method == 'clustering' and not args.agg:
        agent = Cluster(setting=args.setting, data=graph, args=args)
    elif args.method == 'clustering' and args.agg:
        agent = ClusterAgg(setting=args.setting, data=graph, args=args)
    elif args.method == 'averaging':
        agent = Average(setting=args.setting, data=graph, args=args)
    elif args.method == 'gcdm':
        agent = GCDM(setting=args.setting, data=graph, args=args) #currently the code is same as GCDMX
    elif args.method == 'gcdmx':
        agent = GCDMX(setting=args.setting, data=graph, args=args)
    elif args.method == 'gdem':
        agent = GDEM(setting=args.setting, data=graph, args=args)
    elif args.method == 'variation_edges':
        agent = VariationEdges(setting=args.setting, data=graph, args=args)
    elif args.method == 'variation_neighborhoods':
        agent = VariationNeighborhoods(setting=args.setting, data=graph, args=args)
    elif args.method == 'algebraic_JC':
        agent = AlgebraicJc(setting=args.setting, data=graph, args=args)
    elif args.method == 'affinity_GS':
        agent = AffinityGs(setting=args.setting, data=graph, args=args)
    elif args.method == 't_spanner':
        agent = TSpanner(setting=args.setting, data=graph, args=args)
    else:
        agent = eval(to_camel_case(args.method))(setting=args.setting, data=graph, args=args)
    reduced_graph = agent.reduce(graph, verbose=args.verbose)
    evaluator = Evaluator(args)
    res_mean, res_std = evaluator.evaluate(reduced_graph, model_type=args.final_eval_model)
    # args.logger.info(f'Test Mean Accuracy: {100 * all_res[:, 0].mean():.2f} +/- {100 * all_res[:, 1].mean():.2f}')
