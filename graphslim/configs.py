'''Configuration'''
import json
import os

import click

from graphslim.utils import seed_everything


class Obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


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
    args.hidden = 256
    args.checkpoints = range(0, args.epochs + 1, 100)
    args.eval_hidden = 256
    args.eval_epochs = 600
    args.lr_test = 1e-2
    return args


# recommend hyperparameters here
def method_config(args):
    try:
        conf_dt = json.load(open(f'configs/{args.method}/{args.dataset}.json'))
        update_from_dict(args, conf_dt)
    except:
        print('No config file found')
    # little patch for configs
    if args.method in ['doscond'] and args.reduction_rate == 0.1:
        args.outer_loop = 3

    return args


@click.command()
@click.option('--dataset', '-D', default='cora', show_default=True)
@click.option('--gpu_id', default=0, help='gpu id start from 0, -1 means cpu', show_default=True)
@click.option('--setting', '-S', type=click.Choice(['trans', 'ind']), show_default=True)
@click.option('--split', default='fixed', show_default=True)  # 'fixed', 'random', 'few'
@click.option('--run_evaluation', default=10, show_default=True)
@click.option('--run_reduction', default=3, show_default=True)
@click.option('--hidden', '-H', default=256, show_default=True)
@click.option('--eval_hidden', '--eh', default=256, show_default=True)
@click.option('--eval_epochs', '--ee', default=600, show_default=True)
@click.option('--epochs', '--eps', default=1000, show_default=True)
# @click.option('--patience', '-P', default=20, show_default=True)  # only for msgc
@click.option('--lr', default=0.01, show_default=True)
@click.option('--weight_decay', '--wd', default=0, show_default=True)
# @click.option('--normalize_features', is_flag=True, show_default=True)
@click.option('--pre_norm', is_flag=True, show_default=True)
@click.option('--outer_loop', default=10, show_default=True)
@click.option('--inner_loop', default=1, show_default=True)
@click.option('--reduction_rate', '-R', default=0.5, show_default=True, help='reduction rate of training set')
@click.option('--seed', default=1, help='Random seed.', show_default=True)
@click.option('--nlayers', default=2, help='number of GNN layers', show_default=True)
@click.option('--verbose', is_flag=True, show_default=True)
@click.option('--init', default='random', help='initialization synthetic features',
              type=click.Choice(
                  ['random', 'clustering', 'degree', 'pagerank', 'kcenter', 'herding']
              ), show_default=True)
@click.option('--method', '-M', default='kcenter',
              type=click.Choice(
                  ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC',
                   'affinity_GS', 'kron', 'vng', 'clustering', 'averaging',
                   'gcond', 'doscond', 'gcondx', 'doscondx', 'sfgc', 'msgc', 'disco', 'sgdd', 'gcsntk',
                   'cent_d', 'cent_p', 'kcenter', 'herding', 'random']), show_default=True)
@click.option('--activation', default='relu', help='activation function when do NAS',
              type=click.Choice(
                  ['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leakyrelu', 'relu6', 'elu']
              ), show_default=True)
@click.option('--aggpreprocess', is_flag=True, show_default=True)
@click.option('--dis_metric', default='ours', show_default=True)
@click.option('--lr_adj', default=1e-4, show_default=True)
@click.option('--lr_feat', default=1e-4, show_default=True)
@click.option('--lr_test', default=1e-2, show_default=True)
@click.option('--epsilon', default=0, show_default=True, help='sparsificaiton threshold before evaluation')
@click.option('--dropout', default=0.0, show_default=True)
@click.option('--ntrans', default=1, show_default=True)
@click.option('--with_bn', is_flag=True, show_default=True)
@click.option('--batch_adj', default=1, show_default=True, help='batch size for msgc')
# model specific args
@click.option('--alpha', default=0, help='for appnp', show_default=True)
@click.pass_context
def cli(ctx, **kwargs):
    try:
        args = dict2obj(kwargs)
        if args.gpu_id >= 0:
            args.device = f'cuda:{args.gpu_id}'
        else:
            # if gpu_id=-1, use cpu
            args.device = 'cpu'
        # print("device:", args.device)
        seed_everything(args.seed)
        path = "checkpoints/"
        if not os.path.isdir(path):
            os.mkdir(path)
        args.path = path
        # for benchmark, we need unified settings and reduce flexibility of args
        args = method_config(args)
        # setting_config has higher priority than methods_config
        args = setting_config(args)
        return args
    except Exception as e:
        click.echo(f'An error occurred: {e}', err=True)
    # print(args)


if __name__ == '__main__':
    cli()
