'''Configuration'''
import os
import sys
import importlib

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
import json
import logging

import click
from pprint import pformat
import graphslim
from graphslim.utils import seed_everything, f1_macro, accuracy, roc_auc


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
        # set higher priority from command line as we explore some factors
        if key in ['init'] and obj.init is not None:
            continue
        setattr(obj, key, value)


def _stringify_value(value):
    if isinstance(value, logging.Logger):
        return value.name
    if callable(value):
        return getattr(value, '__name__', str(value))
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return f"{value:.6g}"
    value_str = pformat(value, compact=True)
    return " ".join(value_str.split())


PARAMETER_CATEGORIES = {
    'Common': {
        'dataset', 'method', 'gpu_id', 'device', 'setting', 'split', 'run_reduction', 'run_eval',
        'run_inter_eval', 'eval_interval', 'hidden', 'condense_model', 'epochs', 'agg', 'multi_label',
        'dis_metric', 'lr_adj', 'lr_feat', 'optim', 'threshold', 'dropout', 'ntrans', 'with_bn',
        'save_path', 'load_path', 'eval_whole', 'with_structure', 'lr', 'weight_decay', 'pre_norm',
        'outer_loop', 'inner_loop', 'reduction_rate', 'seed', 'nlayers', 'verbose', 'soft_label',
        'init', 'checkpoints', 'logger','metric'
    },
    'Downstream Evaluation': {
        'eval_epochs', 'eval_model', 'final_eval_model', 'eval_wd', 'eval_loss', 'activation', 'alpha'
    },
    'Attack': {
        'ptb_r', 'attack'
    },
    'Coarsening': {
        'coarsen_strategy'
    },
    'MSGC': {
        'batch_adj'
    },
    'GCSNTK': {
        'mx_size'
    },
    'Tspanner': {
        'ts'
    },
    'SimGC': {
        'feat_alpha', 'smoothness_alpha'
    },
    'GDEM': {
        'eigen_k', 'ratio', 'lr_eigenvec', 'gamma', 'test_epochs', 
    },
}

CATEGORY_ORDER = [
    'Common',
    'Downstream Evaluation',
    'Attack',
    'Coarsening',
    'GEOM, SFGC',
    'MSGC',
    'GCSNTK',
    'Tspanner',
    'SimGC',
    'GDEM',
]

COARSENING_METHODS = {
    'variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_jc',
    'affinity_gs', 'kron', 'vng', 'clustering', 'averaging'
}

CATEGORY_METHOD_REQUIREMENTS = {
    'Coarsening': COARSENING_METHODS,
    'GEOM, SFGC': {'geom', 'sfgc'},
    'MSGC': {'msgc'},
    'GCSNTK': {'gcsntk'},
    'Tspanner': {'tspanner', 't_spanner'},
    'SimGC': {'simgc'},
    'GDEM': {'gdem'},
}


def _build_table(rows, headers=('Parameter', 'Value')):
    if not rows:
        return ''
    processed_rows = [headers] + [(key, value) for key, value in rows]
    widths = [max(len(str(row[idx])) for row in processed_rows) for idx in range(len(headers))]

    def _format_row(row):
        return "| " + " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    table_lines = [separator, _format_row(headers), separator]
    for row in rows:
        table_lines.append(_format_row(row))
    table_lines.append(separator)
    return "\n".join(table_lines)


def _get_category_for_key(key):
    if key in PARAMETER_CATEGORIES['Common']:
        return 'Common'
    for category, keys in PARAMETER_CATEGORIES.items():
        if category == 'Common':
            continue
        if key in keys:
            return category
    return None


def _should_include_category(category, args):
    if category == 'Attack':
        attack = getattr(args, 'attack', None)
        return attack not in (None, 'none', 'None')
    if category in CATEGORY_METHOD_REQUIREMENTS:
        method = getattr(args, 'method', None)
        if not isinstance(method, str):
            return False
        return method.lower() in CATEGORY_METHOD_REQUIREMENTS[category]
    return True


def format_args_tables(args):
    args_dict = getattr(args, '__dict__', {})
    if not args_dict:
        return ''
    grouped = {category: [] for category in CATEGORY_ORDER}
    method = getattr(args, 'method', None)
    method_specific_category = None
    if isinstance(method, str):
        method_specific_category = method.upper()
        grouped.setdefault(method_specific_category, [])
    method_config_keys = set(getattr(args, '_method_config_keys', set()))
    for key, value in args_dict.items():
        if key.startswith('_'):
            continue
        category = _get_category_for_key(key)
        if category is None and method_specific_category:
            category = method_specific_category
        if key in method_config_keys and category != 'Common' and method_specific_category:
            category = method_specific_category
        if category is None:
            category = 'Common'
        grouped.setdefault(category, []).append((key, _stringify_value(value)))

    tables = []
    ordered_categories = CATEGORY_ORDER.copy()
    if method_specific_category and method_specific_category not in ordered_categories:
        ordered_categories.insert(2, method_specific_category)
    for category in ordered_categories:
        rows = grouped.get(category, [])
        if not rows:
            continue
        if not _should_include_category(category, args):
            continue
        rows.sort(key=lambda item: item[0])
        table = _build_table(rows)
        if table:
            tables.append(f"{category}\n{table}")

    return "\n\n".join(tables)


# fix setting here
def setting_config(args):
    representative_r = {
        'cora': 0.5,
        'citeseer': 0.5,
        'pubmed': 0.5,
        'flickr': 0.01,
        'reddit': 0.001,
        'ogbn-arxiv': 0.01,
        'yelp': 0.001,
        'amazon': 0.002,
        'pubmed': 0.1
    }
    if args.reduction_rate == -1:
        args.reduction_rate = representative_r[args.dataset]
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'ogbn-arxiv', 'pubmed']:
        args.setting = 'trans'
    if args.dataset in ['flickr', 'reddit', 'amazon','yelp']:
        args.setting = 'ind'
    # args.pre_norm = True
    args.metric = f1_macro if args.dataset in ['yelp', 'amazon'] else accuracy
    # args.metric = 'accuracy'
    args.run_inter_eval = 3
    args.eval_interval = args.epochs // 10
    # if args.method not in ['gcsntk']:
    #     args.eval_interval = max(args.epochs // 10, 1)
    args.checkpoints = list(range(-1, args.epochs + 1, args.eval_interval))
    args.eval_epochs = 300
    return args


# recommend hyperparameters here
def method_config(args):
    try:
        # print(os.path.dirname(graphslim.__file__))
        conf_dt = json.load(
            open(f"{os.path.join(os.path.dirname(graphslim.__file__), 'configs', args.method, args.dataset)}.json"))
        update_from_dict(args, conf_dt)
        existing_keys = getattr(args, '_method_config_keys', set())
        if not isinstance(existing_keys, set):
            existing_keys = set(existing_keys)
        args._method_config_keys = existing_keys.union(set(conf_dt.keys()))
    except:
        print('No config file found or error in json format, please use method_config(args)')
    if args.method in ['msgc']:
        args.batch_adj = 16
        # add temporary changes here
        # do not modify the config json

    return args


@click.command()
#=========common args=====#
@click.option('--dataset', '-D', default='cora', show_default=True)
@click.option('--method', '-M', default='kcenter', show_default=True)
@click.option('--gpu_id', '-G', default=0, help='gpu id start from 0, -1 means cpu', show_default=True)
@click.option('--setting', type=click.Choice(['trans', 'ind']), show_default=True,
              help='transductive or inductive setting')
@click.option('--split', default='fixed', show_default=True,
              help='train/validation/test split, only support public split now, do not change it')  # 'fixed', 'random', 'few'
@click.option('--run_reduction', default=3, show_default=True, help='repeat times of reduction')
@click.option('--run_eval', default=10, show_default=True, help='repeat times of final evaluations')
@click.option('--run_inter_eval', default=5, show_default=True, help='repeat times of intermediate evaluations')
@click.option('--eval_interval', default=100, show_default=True, help='args.eval_interval = args.epochs // 10 by default')
@click.option('--hidden', '-H', default=256, show_default=True, help='number of hidden units in GNN layers')
@click.option('--condense_model', default='SGC',
              type=click.Choice(
                  ['GCN', 'GAT', 'SGC', 'APPNP', 'Cheby', 'GraphSage', 'GAT']
              ), show_default=True)
@click.option('--epochs', '-E', default=1000, show_default=True, help='number of reduction epochs')
@click.option('--agg', is_flag=True, show_default=True, help='use aggregation for coreset methods')
@click.option('--multi_label', is_flag=True, show_default=True, help='multi-label classification')
@click.option('--dis_metric', default='ours', show_default=True,
              help='distance metric for all condensation methods,ours means metric used in GCond paper')
@click.option('--lr_adj', default=1e-4, show_default=True)
@click.option('--lr_feat', default=1e-4, show_default=True)
@click.option('--optim', default="Adam", show_default=True)
@click.option('--threshold', default=0.0, show_default=True, help='sparsificaiton threshold before evaluation')
@click.option('--dropout', default=0.0, show_default=True)
@click.option('--ntrans', default=1, show_default=True, help='number of transformations in SGC and APPNP')
@click.option('--with_bn', is_flag=True, show_default=True)
@click.option('--save_path', '--sp', default='../checkpoints', show_default=True, help='save path for synthetic graph')
@click.option('--load_path', '--lp', default='../../data', show_default=True, help='save path for synthetic graph')
@click.option('--eval_whole', '-W', is_flag=True, show_default=True, help='if run on whole graph')
@click.option('--with_structure', default=1, show_default=True, help='if synthesizing structure')
@click.option('--lr', default=0.01, show_default=True)
@click.option('--weight_decay', '--wd', default=0.0, show_default=True)
@click.option('--pre_norm', default=True, show_default=True,
              help='pre-normalize features, forced true for arxiv, flickr and reddit')
@click.option('--outer_loop', default=10, show_default=True)
@click.option('--inner_loop', default=1, show_default=True)
@click.option('--reduction_rate', '-R', default=-1.0, show_default=True,
              help='-1 means use representative reduction rate; reduction rate of training set, defined as (number of nodes in small graph)/(number of nodes in original graph)')
@click.option('--seed', '-S', default=1, help='Random seed', show_default=True)
@click.option('--nlayers', default=2, help='number of GNN layers of condensed model', show_default=True)
@click.option('--verbose', '-V', is_flag=True, show_default=True)
@click.option('--soft_label', default=0, show_default=True)
@click.option('--init', default='random', help='features initialization methods',
              type=click.Choice(
                  ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC',
                   'affinity_GS', 'kron', 'vng', 'clustering', 'averaging',
                   'cent_d', 'cent_p', 'kcenter', 'herding', 'random']
              ), show_default=True)
#=========downstream evaluation=====#
@click.option('--eval_epochs', '--ee', default=300, show_default=True)
@click.option('--eval_model', '--em', default='GCN',
              type=click.Choice(
                  ['GCN', 'GAT', 'SGC', 'APPNP', 'Cheby', 'GraphSage', 'GAT', 'SGFormer']
              ), show_default=True)
@click.option('--eval_wd', '--ewd', default=0.0, show_default=True)
@click.option('--eval_loss', '--eloss', default='CE',
                type=click.Choice(
                  ['CE', 'KLD','MSE']
              ), show_default=True)
@click.option('--activation', default='relu', help='activation function when do NAS',
              type=click.Choice(
                  ['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leakyrelu', 'relu6', 'elu']
              ), show_default=True)
@click.option('--alpha', default=0.1, help='for downstream appnp', show_default=True)

#=========attack=====#
@click.option('--ptb_r', '-P', default=0.25, show_default=True, help='perturbation rate for corruptions')
@click.option('--attack', '-A', default=None, help='corruption method',
              type=click.Choice(
                  ['random_adj', 'metattack', 'random_feat']
              ), show_default=True)
# ===============coarsening==============#
@click.option('--coarsen_strategy', '--cs', default='greedy', help='for edge contraction method',
              type=click.Choice(
                  ['optimal', 'greedy']
              ), show_default=True)
# ==============GEOM, SFGC ===========================#
@click.option('--no_buff', is_flag=True, show_default=True,
              help='skip the buffer generation and use existing in geom,sfgc')
# ==============MSGC==================================#
@click.option('--batch_adj', default=1, show_default=True, help='batch size for msgc')

#==================gcsntk=====#
@click.option('--mx_size', default=100, help='for gcsntk methods, avoid SVD error', show_default=True)
#==================tspanner=====#
@click.option('--ts', default=4, help='for tspanner', show_default=True)
# ======================================simgc====================================== #
@click.option('--feat_alpha', default=10, show_default=True, help='feature loss weight')
@click.option('--smoothness_alpha', default=0.1, show_default=True, help='smoothness loss weight')
#==================gdem=====#
@click.option('--eigen_k', default=60, show_default=True, help='number of eigenvalues')
@click.option('--ratio', default=0.8, show_default=True, help='eigenvalue loss weight')
@click.option('--lr_eigenvec', default=0.01, show_default=True, help='eigenvalue loss weight')
@click.option('--gamma', default=0.5, show_default=True, help='eigenvalue loss weight')
@click.pass_context
def cli(ctx, **kwargs):
    args = dict2obj(kwargs)
    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        args.device = f'cuda:{args.gpu_id}'
        try:
            torch = sys.modules.get('torch')
            if torch is None:
                torch = importlib.import_module('torch')
            if hasattr(torch, 'cuda'):
                torch.cuda.set_device(args.gpu_id)
        except Exception:
            pass
    else:
        # if gpu_id=-1, use cpu
        args.device = 'cpu'
    path = args.save_path
    # for benchmark, we need unified settings and reduce flexibility of args
    args = method_config(args)
    # setting_config has higher priority than methods_config
    args = setting_config(args)
    for key, value in ctx.params.items():
        if ctx.get_parameter_source(key) == click.core.ParameterSource.COMMANDLINE:
            setattr(args, key, value)
    if not os.path.exists(f'{path}/logs/{args.method}'):
        try:
            os.makedirs(f'{path}/logs/{args.method}')
        except:
            print(f'{path}/logs/{args.method} exists!')
    logging.basicConfig(filename=f'{path}/logs/{args.method}/{args.dataset}_{args.reduction_rate}.log',
                        level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args.logger = logging.getLogger(__name__)
    table = format_args_tables(args)
    if table:
        args.logger.info("Resolved configuration:\n%s", table)
        click.echo(table)
    return args


def get_args():
    return cli(standalone_mode=False)


if __name__ == '__main__':
    cli()
