'''Configuration'''
import click
import json

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
@click.option('--dataset', type=str, default='cora')
@click.option('--gpu_id', type=int, default=0, help='gpu id')
@click.option('--setting', '-S', type=str, default='trans', help='trans/ind')
@click.option('--experiment', type=str, default='fixed')  # 'fixed', 'random', 'few'
@click.option('--runs', type=int, default=10)
@click.option('--hidden', type=int, default=256)
@click.option('--epochs', type=int, default=500)
@click.option('--early_stopping', type=int, default=10)
@click.option('--lr', type=float, default=0.01)
@click.option('--weight_decay', type=float, default=5e-4)
@click.option('--normalize_features', type=bool, default=True)
@click.option('--reduction_rate', type=float, default=0.03)
@click.option('--keep_ratio', type=float, default=1.0)
@click.option('--seed', type=int, default=15, help='Random seed.')
@click.option('--nlayers', type=int, default=2, help='number of GNN layers')
@click.option('--save', type=int, default=0)
@click.option('--method', default='kcenter',
              type=click.Choice(
                  ['vn',
                   'gcond',
                   'kcenter', 'herding', 'random']))
@click.option('--dis_metric', type=str, default='ours')
@click.option('--lr_adj', type=float, default=1e-4)
@click.option('--lr_feat', type=float, default=1e-4)
@click.option('--lr_model', type=float, default=0.01)
@click.option('--dropout', type=float, default=0.0)
@click.option('--alpha', type=float, default=0, help='regularization term.')
@click.option('--debug', type=int, default=0)
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
