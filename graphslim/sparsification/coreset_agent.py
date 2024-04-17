from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from graphslim.dataset.utils import save_reduced
from graphslim.models import *
from graphslim.utils import accuracy, to_tensor


class CoreSet:

    def __init__(self, setting, data, args, **kwargs):
        self.data = data
        self.args = args
        self.setting = setting

        self.device = args.device
        # n = int(data.feat_train.shape[0] * args.reduction_rate)

    def reduce(self, data):
        args = self.args
        if args.method == 'kcenter':
            self.agent = KCenter
        elif args.method == 'herding':
            self.agent = Herding
        elif args.method == 'random':
            self.agent = Random
        else:
            self.agent = None
        model = GCN(nfeat=data.feat_full.shape[1], nhid=args.hidden, nclass=data.nclass, device=args.device,
                    weight_decay=args.weight_decay).to(args.device)
        if self.setting == 'trans':
            model.fit_with_val(data, train_iters=600, verbose=True, setting='trans')
            # model.test(data, verbose=True)
            embeds = model.predict(data.feat_full, data.adj_full).detach()

            idx_selected = self.agent(embeds, data, args)

            # induce a graph with selected nodes
            data.feat_syn = data.feat_full[idx_selected]
            data.adj_syn = data.adj_full[np.ix_(idx_selected, idx_selected)]

            data.labels_syn = data.labels_full[idx_selected]

            # if args.save:
            #     np.save(f'dataset/output/coreset/idx_{args.dataset}_{args.reduction_rate}_{args.method}_{args.seed}.npy',
            #             idx_selected)

            # for i in tqdm(range(args.runs)):
            #     seed_everything(args.seed + i)
            #     model.fit_with_val(feat_selected, adj_selected, data,
            #                        train_iters=args.epochs, normalize=True, verbose=False, reindexed_trainset=True,
            #                        reduced=True)
            #
            #     # Full graph
            #     # interface: model.test(full_data)
            #     acc_test = model.test(data)
            #     res.append(acc_test)

        if self.setting == 'ind':
            model.fit_with_val(data, train_iters=600, normalize=True,
                               verbose=True, setting='ind', reindex=True)

            model.eval()

            embeds = model.predict(data.feat_train, data.adj_train).detach()

            idx_selected = self.agent(embeds, data, args)

            data.feat_syn = data.feat_train[idx_selected]
            data.adj_syn = data.adj_train[np.ix_(idx_selected, idx_selected)]

            data.labels_syn = data.labels_train[idx_selected]

            # for i in tqdm(range(args.runs)):
            #     seed_everything(args.seed + i)
            #     model.fit_with_val(feat_selected, adj_selected, data,
            #                        train_iters=args.epochs, normalize=True, verbose=False, val=True, reduced=True)
            #
            #     model.eval()
            #     labels_test = torch.LongTensor(data.labels_test).cuda()
            #
            #     # interface: model.predict(reshaped feat,reshaped adj)
            #     output = model.predict(feat_test, adj_test)
            #     # loss_test = F.nll_loss(output, labels_test)
            #     acc_test = accuracy(output, labels_test)
            #     res.append(acc_test.item())
        data.adj_syn, data.feat_syn, data.labels_syn = to_tensor(data.adj_syn, data.feat_syn, data.labels_syn,
                                                                 device='cpu')
        if args.save:
            save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)

        return data


def prepare_select(data, args):
    num_class_dict = {}
    syn_class_indices = {}
    if args.setting == 'ind':
        idx_train = np.arange(len(data.idx_train))
    else:
        idx_train = data.idx_train
    labels_train = data.labels_train
    # d = data.feat_train.shape[1]
    counter = Counter(data.labels_train.tolist())
    # n = len(data.labels_train)
    sorted_counter = sorted(counter.items(), key=lambda x: x[1])
    sum_ = 0
    labels_syn = []
    for ix, (c, num) in enumerate(sorted_counter):
        num_class_dict[c] = max(int(num * args.reduction_rate), 1)
        sum_ += num_class_dict[c]
        syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
        labels_syn += [c] * num_class_dict[c]

    return num_class_dict, labels_train, idx_train


def KCenter(embeds, data, args):
    # feature: embeds
    # kcenter # class by class
    num_class_dict, labels_train, idx_train = prepare_select(data, args)
    idx_selected = []
    for class_id, cnt in num_class_dict.items():
        idx = idx_train[labels_train == class_id]
        feature = embeds[idx]
        mean = torch.mean(feature, dim=0, keepdim=True)
        # dis = distance(feature, mean)[:,0]
        dis = torch.cdist(feature, mean)[:, 0]
        rank = torch.argsort(dis)
        idx_centers = rank[:1].tolist()
        for i in range(cnt - 1):
            feature_centers = feature[idx_centers]
            dis_center = torch.cdist(feature, feature_centers)
            dis_min, _ = torch.min(dis_center, dim=-1)
            id_max = torch.argmax(dis_min).item()
            idx_centers.append(id_max)

        idx_selected.append(idx[idx_centers])
    # return np.array(idx_selected).reshape(-1)
    return np.hstack(idx_selected)


def Herding(embeds, data, args):
    num_class_dict, labels_train, idx_train = prepare_select(data, args)

    # herding # class by class
    idx_selected = []
    for class_id, cnt in num_class_dict.items():
        idx = idx_train[labels_train == class_id]
        features = embeds[idx]
        mean = torch.mean(features, dim=0, keepdim=True)
        selected = []
        idx_left = np.arange(features.shape[0]).tolist()

        for i in range(cnt):
            det = mean * (i + 1) - torch.sum(features[selected], dim=0)
            dis = torch.cdist(det, features[idx_left])
            id_min = torch.argmin(dis)
            selected.append(idx_left[id_min])
            del idx_left[id_min]
        idx_selected.append(idx[selected])
    # return np.array(idx_selected).reshape(-1)
    return np.hstack(idx_selected)


def Random(embeds, data, args):
    num_class_dict, labels_train, idx_train = prepare_select(data, args)
    idx_selected = []

    for class_id, cnt in num_class_dict.items():
        idx = idx_train[labels_train == class_id]
        selected = np.random.permutation(idx)
        idx_selected.append(selected[:cnt])

    # return np.array(idx_selected).reshape(-1)
    return np.hstack(idx_selected)
