import argparse
from coarsening import *
from dataset import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--gpu_id', type=int, default=2, help='gpu id')
# TODO: implement setting
parser.add_argument('--setting', '-S', type=str, default='trans', help='trans/ind')
parser.add_argument('--experiment', type=str, default='fixed')  # 'fixed', 'random', 'few'
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--normalize_features', type=bool, default=True)
# remind: 0.026 equals 0.5 of training set (Cora) 0.018 equals 0.5 (citeseer) 0.003 equals 0.5 (pubmed)
parser.add_argument('--reduction_rate', type=float, default=0.03)
parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
args = parser.parse_args()
print(args)

path = "checkpoints/"
if not os.path.isdir(path):
    os.mkdir(path)

torch.cuda.set_device(args.gpu_id)

data = get_dataset(args.dataset, return_pyg=True)

agent = CoarseningBase(data, args, device='cuda')

agent.train()


