from collections import Counter

import torch.nn as nn

from graphslim.condensation.gcond_base import GCondBase
from graphslim.condensation.utils import match_loss
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation import *
from graphslim.models import *
from graphslim.utils import *
from tqdm import trange


class SGDD(GCondBase):
    def __init__(self, setting, data, args, **kwargs):
        super(SGDD, self).__init__(setting, data, args, **kwargs)

        self.pge = IGNR(node_feature=self.d, nfeat=128, nnodes=self.nnodes_syn, device=self.device, args=args
                        ).to(self.device)
        # self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        print('adj_syn:', (self.nnodes_syn, self.nnodes_syn), 'feat_syn:', self.feat_syn.shape)


    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args

        if data.adj_full.shape[0] < args.mx_size:
            args.mx_size = data.adj_full.shape[0]
        else:
            data.adj_mx = data.adj_full[: args.mx_size, : args.mx_size]

        feat_syn, labels_syn = to_tensor(self.feat_syn, label=data.labels_syn, device=self.device)
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train,
                                              device=self.device)

        # initialization the features
        feat_init = self.init()
        self.feat_syn.data.copy_(feat_init)

        adj = normalize_adj_tensor(adj, sparse=True)

        outer_loop, inner_loop = self.get_loops(args)
        loss_avg = 0
        best_val = 0

        model = eval(args.condense_model)(feat_syn.shape[1], args.hidden,
                                          data.nclass, args).to(self.device)
        for it in trange(args.epochs):

            model.initialize()
            model_parameters = list(model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr)
            model.train()

            for ol in range(outer_loop):
                adj_syn, opt_loss = self.pge(self.feat_syn, Lx=data.adj_mx)
                self.adj_syn = normalize_adj_tensor(adj_syn, sparse=False)

                model = self.check_bn(model)
                loss = self.train_class(model, adj, features, labels, labels_syn, args)
                if args.opt_scale > 0:
                    loss_opt = args.opt_scale * opt_loss
                else:
                    loss_opt = torch.tensor(0)

                loss = loss + loss_opt
                loss_avg += loss.item()

                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()

                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()
                feat_syn_inner = self.feat_syn.detach()
                adj_syn_inner = self.pge.inference(feat_syn_inner)
                adj_syn_inner_norm = normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    optimizer_model.step()

            loss_avg /= (data.nclass * outer_loop)

            if it in args.checkpoints:
                self.adj_syn = adj_syn_inner
                data.adj_syn, data.feat_syn, data.labels_syn = self.adj_syn.detach(), self.feat_syn.detach(), labels_syn.detach()
                best_val = self.intermediate_evaluation(best_val, loss_avg)

        return data

    def sampling(self, ids_per_cls_train, budget, vecs, d, using_half=True):
        budget_dist_compute = 1000
        '''
        if using_half:
            vecs = vecs.half()
        '''
        if isinstance(vecs, np.ndarray):
            vecs = torch.from_numpy(vecs)
        vecs = vecs.half()
        ids_selected = []
        for i, ids in enumerate(ids_per_cls_train):
            class_ = list(budget.keys())[i]
            other_cls_ids = list(range(len(ids_per_cls_train)))
            other_cls_ids.pop(i)
            ids_selected0 = ids_per_cls_train[i] if len(ids_per_cls_train[i]) < budget_dist_compute else random.choices(
                ids_per_cls_train[i], k=budget_dist_compute)

            dist = []
            vecs_0 = vecs[ids_selected0]
            for j in other_cls_ids:
                chosen_ids = random.choices(ids_per_cls_train[j], k=min(budget_dist_compute, len(ids_per_cls_train[j])))
                vecs_1 = vecs[chosen_ids]
                if len(chosen_ids) < 26 or len(ids_selected0) < 26:
                    # torch.cdist throws error for tensor smaller than 26
                    dist.append(torch.cdist(vecs_0.float(), vecs_1.float()).half())
                else:
                    dist.append(torch.cdist(vecs_0, vecs_1))

            # dist = [torch.cdist(vecs[ids_selected0], vecs[random.choices(ids_per_cls_train[j], k=min(budget_dist_compute,len(ids_per_cls_train[j])))]) for j in other_cls_ids]
            dist_ = torch.cat(dist, dim=-1)  # include distance to all the other classes
            n_selected = (dist_ < d).sum(dim=-1)
            rank = n_selected.sort()[1].tolist()
            current_ids_selected = rank[:budget[class_]] if len(rank) > budget[class_] else random.choices(rank,
                                                                                                           k=budget[
                                                                                                               class_])
            ids_selected.extend([ids_per_cls_train[i][j] for j in current_ids_selected])
        return ids_selected

    def get_sub_adj_feat(self, features):
        data = self.data
        args = self.args
        idx_selected = []

        counter = Counter(self.data.labels_syn)
        labels_train = self.data.labels_train.squeeze().tolist()  # important
        ids_per_cls_train = [(labels_train == c).nonzero()[0] for c in counter.keys()]
        idx_selected = self.sampling(ids_per_cls_train, counter, features, 0.5, counter)
        features = features[idx_selected]

        return features, None
