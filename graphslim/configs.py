'''Configuration'''
import json
import os
import logging

import click
from pprint import pformat
from graphslim.utils import seed_everything


class Obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

    def __repr__(self):
        # Use pprint's pformat to print the dictionary in a pretty manner
        return pformat(self.__dict__, compact=True)


def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=Obj)


def update_from_dict(obj, updates):
    for key, value in updates.items():
        setattr(obj, key, value)


# fix setting here
def setting_config(args):
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'ogbn-arxiv']:
        args.setting = 'trans'
    if args.dataset in ['flickr', 'reddit']:
        args.setting = 'ind'
    args.pre_norm = True
    args.run_inter_eval = 3
    if args.method not in ['gcsntk']:
        args.eval_interval = max(args.epochs // 10, 1)
    args.checkpoints = list(range(-1, args.epochs + 1, args.eval_interval))
    args.eval_epochs = 300
    return args


# recommend hyperparameters here
def method_config(args):
    try:
        conf_dt = json.load(open(f'configs/{args.method}/{args.dataset}.json'))
        update_from_dict(args, conf_dt)
    except:
        print('No config file found or error in json format.')
    if args.method in ['msgc']:
        args.batch_adj = 16
        # add temporary changes here
        # do not modify the config json

    return args


@click.command()
@click.option('--dataset', '-D', default='cora', show_default=True)
@click.option('--gpu_id', '-G', default=0, help='gpu id start from 0, -1 means cpu', show_default=True)
@click.option('--setting', type=click.Choice(['trans', 'ind']), show_default=True)
@click.option('--split', default='fixed', show_default=True)  # 'fixed', 'random', 'few'
@click.option('--run_eval', default=10, show_default=True)
@click.option('--run_inter_eval', default=5, show_default=True)
@click.option('--run_reduction', default=3, show_default=True)
@click.option('--eval_interval', default=100, show_default=True)
@click.option('--hidden', '-H', default=256, show_default=True)
@click.option('--eval_epochs', '--ee', default=300, show_default=True)
@click.option('--eval_model', default='GCN',
              type=click.Choice(
                  ['GCN', 'GAT', 'SGC', 'APPNP', 'Cheby', 'GraphSage', 'GAT']
              ), show_default=True)
@click.option('--condense_model', default='SGC',
              type=click.Choice(
                  ['GCN', 'GAT', 'SGC', 'APPNP', 'Cheby', 'GraphSage', 'GAT']
              ), show_default=True)
@click.option('--epochs', '-E', default=1000, show_default=True)
# @click.option('--valid_result', '--vr', default=0.0, show_default=True)
# @click.option('--patience', '-P', default=20, show_default=True)  # only for msgc
@click.option('--lr', default=0.01, show_default=True)
@click.option('--weight_decay', '--wd', default=0, show_default=True)
# @click.option('--normalize_features', is_flag=True, show_default=True)
@click.option('--pre_norm', is_flag=True, show_default=True)
@click.option('--outer_loop', default=10, show_default=True)
@click.option('--inner_loop', default=1, show_default=True)
@click.option('--reduction_rate', '-R', default=0.5, show_default=True, help='reduction rate of training set')
@click.option('--seed', '-S', default=1, help='Random seed.', show_default=True)
@click.option('--nlayers', default=2, help='number of GNN layers', show_default=True)
@click.option('--verbose', is_flag=True, show_default=True)
@click.option('--init', default='random', help='initialization synthetic features',
              type=click.Choice(
                  ['random', 'clustering', 'averaging', 'kcenter', 'herding']
              ), show_default=True)
@click.option('--method', '-M', default='kcenter',
              type=click.Choice(
                  ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC',
                   'affinity_GS', 'kron', 'vng', 'clustering', 'averaging',
                   'gcond', 'doscond', 'gcondx', 'doscondx', 'sfgc', 'msgc', 'disco', 'sgdd', 'gcsntk', 'geom',
                   'cent_d', 'cent_p', 'kcenter', 'herding', 'random']), show_default=True)
@click.option('--activation', default='relu', help='activation function when do NAS',
              type=click.Choice(
                  ['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leakyrelu', 'relu6', 'elu']
              ), show_default=True)
@click.option('--attack', '-A', default='none', help='attack method',
              type=click.Choice(
                  ['none', 'random', 'dice', 'metattack']
              ), show_default=True)
@click.option('--aggpreprocess', is_flag=True, show_default=True)
@click.option('--dis_metric', default='ours', show_default=True)
@click.option('--lr_adj', default=1e-4, show_default=True)
@click.option('--lr_feat', default=1e-4, show_default=True)
@click.option('--threshold', default=0, show_default=True, help='sparsificaiton threshold before evaluation')
@click.option('--dropout', default=0.0, show_default=True)
@click.option('--ntrans', default=1, show_default=True, help='number of transformations in SGC and APPNP')
@click.option('--with_bn', is_flag=True, show_default=True)
@click.option('--no_buff', is_flag=True, show_default=True, help='skip the buffer in sfgc')
@click.option('--batch_adj', default=1, show_default=True, help='batch size for msgc')
# model specific args
@click.option('--alpha', default=0.1, help='for appnp', show_default=True)
@click.option('--mx_size', default=100, help='for ntk methods, avoid SVD error', show_default=True)
@click.option('-origin', '-O', is_flag=True, help='original or condensed', show_default=True)
@click.option('--save_path', '--sp', default='checkpoints', show_default=True)
@click.option('--ptb_r', '-P', default=0.25, show_default=True)
@click.pass_context
def cli(ctx, **kwargs):
    args = dict2obj(kwargs)
    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
        args.device = f'cuda:0'
    else:
        # if gpu_id=-1, use cpu
        args.device = 'cpu'
    seed_everything(args.seed)
    path = args.save_path
    # for benchmark, we need unified settings and reduce flexibility of args
    args = method_config(args)
    # setting_config has higher priority than methods_config
    args = setting_config(args)
    if not os.path.exists(f'{path}/logs/{args.method}'):
        os.makedirs(f'{path}/logs/{args.method}')
    logging.basicConfig(filename=f'{path}/logs/{args.method}/{args.dataset}_{args.reduction_rate}.log',
                        level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args.logger = logging.getLogger(__name__)
    args.logger.addHandler(logging.StreamHandler())
    args.logger.info(args)
    return args


if __name__ == '__main__':
    cli()
