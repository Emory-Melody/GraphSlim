'''Configuration'''
import json
import os

import click

from utils import seed_everything


class Obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=Obj)


# recommended hyperparameters here
def load_config(args):
    dataset = args.dataset
    if dataset in ['flickr']:
        args.nlayers = 2
        args.hidden = 256
        args.weight_decay = 5e-3
        args.dropout = 0.0

    if dataset in ['reddit']:
        args.nlayers = 2
        args.hidden = 256
        args.weight_decay = 0e-4
        args.dropout = 0

    if dataset in ['ogbn-arxiv']:
        args.hidden = 256
        args.weight_decay = 0
        args.dropout = 0

    return args


@click.command()
@click.option('--dataset', '-D', default='cora', show_default=True)
@click.option('--gpu_id', default=0, help='gpu id start from 0, -1 means cpu', show_default=True)
@click.option('--setting', '-S', type=click.Choice(['trans', 'ind']), show_default=True)
@click.option('--split', default='fixed', show_default=True)  # 'fixed', 'random', 'few'
@click.option('--runs', default=10, show_default=True)
@click.option('--hidden', '-H', default=256, show_default=True)
@click.option('--epochs', '--eps', default=400, show_default=True)
@click.option('--early_stopping', '-E', default=10, show_default=True)
@click.option('--lr', default=0.01, show_default=True)
@click.option('--weight_decay', '--wd', default=5e-4, show_default=True)
@click.option('--normalize_features', '--normalize', is_flag=True, show_default=True)
@click.option('--reduction_rate', '-R', default=0.5, show_default=True, help='reduction rate of training set')
@click.option('--seed', default=42, help='Random seed.', show_default=True)
@click.option('--nlayers', default=2, help='number of GNN layers', show_default=True)
@click.option('--save', is_flag=True, show_default=True)
@click.option('--verbose', is_flag=True, show_default=True)
@click.option('--method', '-M', default='kcenter',
              type=click.Choice(
                  ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC',
                   'affinity_GS', 'kron',
                   'gcond',
                   'kcenter', 'herding', 'random']), show_default=True)
@click.option('--dis_metric', default='ours', show_default=True)
@click.option('--lr_adj', default=1e-4, show_default=True)
@click.option('--lr_feat', default=1e-4, show_default=True)
@click.option('--one_step', is_flag=True, show_default=True)
@click.option('--dropout', default=0.0, show_default=True)
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
        seed_everything(args.seed)
        path = "checkpoints/"
        if not os.path.isdir(path):
            os.mkdir(path)
        args.path = path
        args = load_config(args)
        return args
    except Exception as e:
        click.echo(f'An error occurred: {e}', err=True)
    # print(args)


if __name__ == '__main__':
    cli()
