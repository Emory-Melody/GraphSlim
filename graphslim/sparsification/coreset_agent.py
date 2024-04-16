from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from graphslim.dataset.utils import save_reduced
from graphslim.models import *
from graphslim.utils import accuracy, seed_everything


def router_select(data, args):
    '''
    workflows routing to each method and different settings
    '''
    if args.method == 'kcenter':
        agent = KCenter(data, args)
    elif args.method == 'herding':
        agent = Herding(data, args)
    elif args.method == 'random':
        agent = Random(data, args)
    else:
        pass

    res = []
    model = GCN(nfeat=data.feat_full.shape[1], nhid=args.hidden, nclass=data.nclass, device=args.device,
                weight_decay=args.weight_decay).to(args.device)
    if args.setting == 'trans':
        model.fit_with_val(data.feat_full, data.adj_full, data, train_iters=args.epochs, verbose=True)
        model.test(data, verbose=True)
        embeds = model.predict().detach()

        idx_selected = agent.select(embeds)

        # induce a graph with selected nodes
        feat_selected = data.feat_full[idx_selected]
        adj_selected = data.adj_full[np.ix_(idx_selected, idx_selected)]

        data.labels_syn = data.labels_full[idx_selected]

        # if args.save:
        #     np.save(f'dataset/output/coreset/idx_{args.dataset}_{args.reduction_rate}_{args.method}_{args.seed}.npy',
        #             idx_selected)

        for i in tqdm(range(args.runs)):
            seed_everything(args.seed + i)
            model.fit_with_val(feat_selected, adj_selected, data,
                               train_iters=args.epochs, normalize=True, verbose=False, reindexed_trainset=True,
                               reduced=True)

            # Full graph
            # interface: model.test(full_data)
            acc_test = model.test(data)
            res.append(acc_test)

    if args.setting == 'ind':

        model.fit_with_val(data.feat_train, data.adj_train, data, train_iters=args.epochs, normalize=True,
                           verbose=False, reindexed_trainset=True)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()
        feat_test, adj_test = data.feat_test, data.adj_test

        embeds = model.predict().detach()

        output = model.predict(feat_test, adj_test)
        loss_test = F.nll_loss(output, labels_test)
        acc_test = accuracy(output, labels_test)
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        idx_selected = agent.select(embeds, inductive=True)

        feat_selected = data.feat_train[idx_selected]
        adj_selected = data.adj_train[np.ix_(idx_selected, idx_selected)]

        data.labels_syn = data.labels_train[idx_selected]
        if args.save:
            save_reduced(adj_selected, feat_selected, data.labels_syn, args)

        for i in tqdm(range(args.runs)):
            seed_everything(args.seed + i)
            model.fit_with_val(feat_selected, adj_selected, data,
                               train_iters=args.epochs, normalize=True, verbose=False, val=True, reduced=True)

            model.eval()
            labels_test = torch.LongTensor(data.labels_test).cuda()

            # interface: model.predict(reshaped feat,reshaped adj)
            output = model.predict(feat_test, adj_test)
            # loss_test = F.nll_loss(output, labels_test)
            acc_test = accuracy(output, labels_test)
            res.append(acc_test.item())
    res = np.array(res)
    print('Mean accuracy:', repr([res.mean(), res.std()]))
    return idx_selected


class Base:

    def __init__(self, data, args, **kwargs):
        self.data = data
        self.args = args
        self.device = args.device
        # n = int(data.feat_train.shape[0] * args.reduction_rate)
        d = data.feat_train.shape[1]
        # self.nnodes_syn = n
        self.num_class_dict = {}
        self.syn_class_indices = {}
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(self.device)
        n = len(self.labels_syn)
        print(f'selected graph size: [{n},{n}]')

    def generate_labels_syn(self, data):
        counter = Counter(data.labels_train)
        # n = len(data.labels_train)
        num_class_dict = self.num_class_dict
        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []
        for ix, (c, num) in enumerate(sorted_counter):
            # if ix == len(sorted_counter) - 1:
            #     num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
            #     self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            #     labels_syn += [c] * num_class_dict[c]
            # else:
            num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
            sum_ += num_class_dict[c]
            self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]

        return labels_syn


class KCenter(Base):
    def __init__(self, data, args, **kwargs):
        super().__init__(data, args, **kwargs)

    def select(self, embeds, inductive=False):
        # feature: embeds
        # kcenter # class by class
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train
        labels_train = self.data.labels_train
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


class Herding(Base):
    def __init__(self, data, args, **kwargs):
        super().__init__(data, args, **kwargs)

    def select(self, embeds, inductive=False):
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train
        labels_train = self.data.labels_train
        idx_selected = []

        # herding # class by class
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


class Random(Base):
    def __init__(self, data, args, **kwargs):
        super().__init__(data, args, **kwargs)

    def select(self, embed, inductive=False):
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train

        labels_train = self.data.labels_train
        idx_selected = []

        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train == class_id]
            selected = np.random.permutation(idx)
            idx_selected.append(selected[:cnt])

        # return np.array(idx_selected).reshape(-1)
        return np.hstack(idx_selected)
