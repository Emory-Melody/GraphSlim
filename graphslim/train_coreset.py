import argparse

from tqdm import tqdm

from configs import load_config
from dataset import *
from models.gcn import GCN
from sparsification import KCenter

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--setting', '-S', type=str, default='trans', help='trans/ind')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--nlayers', type=int, default=2, help='Random seed.')
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--method', type=str, default='kcenter', choices=['kcenter', 'herding', 'random'])
parser.add_argument('--reduction_rate', type=float, default=0.5)
args = parser.parse_args()

if args.gpu_id is not None:
    args.device = f'cuda:{args.gpu_id}'
else:
    args.device = 'cpu'
args = load_config(args)
print(args)

seed_everything(args.seed)

data = get_dataset(args.dataset, args.normalize_features)

num_classes = data.labels_full.max() + 1
agent = KCenter(data, args)

model = GCN(nfeat=data.feat_full.shape[1], nhid=args.hidden, nclass=num_classes, device=args.device,
            weight_decay=args.weight_decay).to(args.device)

# ============start==================#
if args.setting == 'trans':
    features = data.feat_full
    adj = data.adj_full
    labels = data.labels_full
    idx_train = data.idx_train
    idx_val = data.idx_val
    idx_test = data.idx_test

    # Setup GCN Model

    model.fit_with_val(features, adj, data, train_iters=args.epochs, verbose=True)
    model.test(data, verbose=True)
    embeds = model.predict().detach()

    idx_selected = agent.select(embeds)

    # induce a graph with selected nodes
    feat_train = features[idx_selected]
    adj_train = adj[np.ix_(idx_selected, idx_selected)]

    data.labels_syn = labels[idx_selected]

    if args.save:
        np.save(f'dataset/output/coreset/idx_{args.dataset}_{args.reduction_rate}_{args.method}_{args.seed}.npy',
                idx_selected)

    res = []
    runs = 10
    for _ in tqdm(range(runs)):
        model.fit_with_val(feat_train, adj_train, data,
                           train_iters=args.epochs, normalize=True, verbose=False, condensed=True)

        # Full graph
        # interface: model.test(full_data)
        acc_test = model.test(data)
        res.append(acc_test)

    res = np.array(res)
    print('Mean accuracy:', repr([res.mean(), res.std()]))

if args.setting == 'ind':
    feat_train = data.feat_train
    adj_train = data.adj_train
    labels_train = data.labels_train

    # Setup GCN Model

    model.fit_with_val(feat_train, adj_train, data, train_iters=args.epochs, normalize=True, verbose=False)

    model.eval()
    labels_test = torch.LongTensor(data.labels_test).cuda()
    feat_test, adj_test = data.feat_test, data.adj_test

    embeds = model.predict().detach()

    output = model.predict(feat_test, adj_test)
    loss_test = F.nll_loss(output, labels_test)
    acc_test = utils.accuracy(output, labels_test)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    idx_selected = agent.select(embeds, inductive=True)

    feat_train = feat_train[idx_selected]
    adj_train = adj_train[np.ix_(idx_selected, idx_selected)]

    data.labels_syn = labels_train[idx_selected]

    res = []
    runs = 10
    for _ in tqdm(range(runs)):
        model.fit_with_val(feat_train, adj_train, data,
                           train_iters=args.epochs, normalize=True, verbose=False, val=True, condensed=True)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        # interface: model.predict(reshaped feat,reshaped adj)
        output = model.predict(feat_test, adj_test)
        loss_test = F.nll_loss(output, labels_test)
        acc_test = utils.accuracy(output, labels_test)
        res.append(acc_test.item())
    res = np.array(res)
    print('Mean accuracy:', repr([res.mean(), res.std()]))
