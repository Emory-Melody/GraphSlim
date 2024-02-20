import argparse
from dataset import *
import deeprobust.graph.utils as utils
from configs import load_config
from utils import *
from dataset import DataGraphSAINT
from models.gcn import GCN
from sparsification.coreset import KCenter, Herding, Random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--nlayers', type=int, default=2, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--inductive', type=int, default=1)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--method', type=str, choices=['kcenter', 'herding', 'random'])
parser.add_argument('--reduction_rate', type=float, required=True)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
args = load_config(args)
print(args)

# # random seed setting
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
#
# data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
# if args.dataset in data_graphsaint:
#     data = DataGraphSAINT(args.dataset)
#     data_full = data.data_full
# else:
#     data_full = get_dataset(args.dataset, args.normalize_features)
#     data = TransAndInd(data_full, keep_ratio=args.keep_ratio)


