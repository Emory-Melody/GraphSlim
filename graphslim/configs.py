'''Configuration'''
import json

import click
from click import Choice

from utils import seed_everything


class Obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=Obj)


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
@click.option('--gpu_id', type=int, default=0, help='gpu id')
@click.option('--setting', '-S', type=str, default='trans', help='trans/ind')
@click.option('--dataset', default='cora')
@click.option('--hidden', type=int, default=64)
@click.option('--normalize_features', type=bool, default=True)
@click.option('--keep_ratio', type=float, default=1.0)
@click.option('--lr', type=float, default=0.01)
@click.option('--weight_decay', type=float, default=5e-4)
@click.option('--seed', type=int, default=15, help='Random seed.')
@click.option('--nlayers', type=int, default=2, help='number of GNN layers')
@click.option('--epochs', type=int, default=600)
@click.option('--save', type=int, default=0)
@click.option('--method', default='kcenter', type=Choice(['gcond', 'kcenter', 'herding', 'random']))
@click.option('--reduction_rate', type=float, default=0.5)
@click.option('--dis_metric', type=str, default='ours')
@click.option('--lr_adj', type=float, default=1e-4)
@click.option('--lr_feat', type=float, default=1e-4)
@click.option('--lr_model', type=float, default=0.01)
@click.option('--dropout', type=float, default=0.0)
@click.option('--alpha', type=float, default=0, help='regularization term.')
@click.option('--debug', type=int, default=0)
@click.option('--sgc', type=int, default=1)
@click.option('--inner', type=int, default=0)
@click.option('--outer', type=int, default=20)
@click.option('--one_step', type=int, default=0)
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
        return args
    except Exception as e:
        click.echo(f'An error occurred: {e}', err=True)
    # print(args)
