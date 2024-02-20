import argparse
from utils import *
from dataset import *
from condensation import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--setting', '-S', type=str, default='trans', help='trans/ind')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--one_step', type=int, default=0)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# random seed setting
seed_everything(args.seed)
print(args)

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    # arxiv: transductive
    # flickr, reddict: inductive
else:
    data = get_dataset(args.dataset, args.normalize_features)
    # trans or ind is optional for cora, citeseer, pubmed
    data = TransAndInd(data, keep_ratio=args.keep_ratio)

if args.setting == 'trans':
    agent = GCondTrans(data, args, device='cuda')
elif args.setting == 'ind':
    agent = GCondInd(data, args, device='cuda')
else:
    raise Exception('Unknown setting')

agent.train()
