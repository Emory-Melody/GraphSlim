from collections import Counter

import torch.nn as nn

from graphslim.condensation.gcond_base import GCondBase
from graphslim.condensation.utils import match_loss
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation import *
from graphslim.models import *
from graphslim.utils import *


class SGDD(GCondBase):
    def __init__(self, setting, data, args, **kwargs):
        # super(SGDD, self).__init__(setting, data, args, **kwargs)
        self.data = data
        self.args = args
        self.device = args.device
        self.setting = setting

        self.data.labels_syn = self.generate_labels_syn(data)
        self.nnodes_syn = len(self.data.labels_syn)
        self.d = data.feat_train.shape[1]

        print(f'target reduced size:{int(data.feat_train.shape[0] * args.reduction_rate)}')
        print(f'actual reduced size:{self.nnodes_syn}')

        self.feat_syn = nn.Parameter(torch.FloatTensor(self.nnodes_syn, self.d).to(self.device))
        self.pge = IGNR(node_feature=self.d, nfeat=128, nnodes=self.nnodes_syn, device=self.device, args=args
                        ).to(self.device)
        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        print('adj_syn:', (self.nnodes_syn, self.nnodes_syn), 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))
        # self.pge.reset_parameters()

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args

        if data.adj_full.shape[0] < args.mx_size:
            args.mx_size = data.adj_full.shape[0]
        else:
            data.adj_mx = data.adj_full[: args.mx_size, : args.mx_size]

        feat_syn, pge, labels_syn = to_tensor(self.feat_syn, self.pge, label=data.labels_syn, device=self.device)
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train,
                                              device=self.device)

        syn_class_indices = self.syn_class_indices

        # initialization the features
        feat_sub, adj_sub = self.get_sub_adj_feat(features)
        self.feat_syn.data.copy_(feat_sub)

        adj = normalize_adj_tensor(adj, sparse=True)

        outer_loop, inner_loop = self.get_loops(args)
        loss_avg = 0
        best_val = 0

        for it in range(args.epochs + 1):
            # seed_everything(args.seed + it)
            if args.dataset in ['ogbn-arxiv', 'flickr']:
                model = SGCRich(nfeat=feat_syn.shape[1], nhid=args.hidden,
                                dropout=0.0, with_bn=False,
                                weight_decay=0e-4, nlayers=args.nlayers,
                                nclass=data.nclass,
                                device=self.device).to(self.device)
            else:
                model = SGC(nfeat=feat_syn.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=0, weight_decay=0,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)

            model.initialize()
            model_parameters = list(model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr)
            model.train()

            for ol in range(outer_loop):
                adj_syn, opt_loss = self.pge(self.feat_syn, Lx=data.adj_mx)

                adj_syn_norm = normalize_adj_tensor(adj_syn, sparse=False)
                # feat_syn_norm = feat_syn

                model = self.check_bn(model)

                loss = torch.tensor(0.0).to(self.device)
                for c in range(data.nclass):
                    batch_size, n_id, adjs = data.retrieve_class_sampler(
                        c, adj, args)

                    if args.nlayers == 1:
                        adjs = [adjs]

                    adjs = [adj.to(self.device) for adj in adjs]
                    output = model.forward_sampler(features[n_id], adjs)
                    loss_real = F.nll_loss(output, labels[n_id[:batch_size]])

                    gw_real = torch.autograd.grad(loss_real, model_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    output_syn = model.forward(feat_syn, adj_syn_norm)

                    ind = syn_class_indices[c]
                    loss_syn = F.nll_loss(
                        output_syn[ind[0]: ind[1]],
                        labels_syn[ind[0]: ind[1]])
                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss += coeff * match_loss(gw_syn, gw_real, args, device=self.device)

                loss_avg += loss.item()

                if args.alpha > 0:
                    loss_reg = args.alpha * regularization(adj_syn, tensor2onehot(labels_syn))
                else:
                    loss_reg = torch.tensor(0)

                if args.opt_scale > 0:
                    loss_opt = args.opt_scale * opt_loss
                else:
                    loss_opt = torch.tensor(0)

                loss = loss + loss_reg + loss_opt

                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()

                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()
                # else:
                #     if ol < outer_loop // 5:
                #         self.optimizer_pge.step()
                #     else:
                #         self.optimizer_feat.step()

                # if verbose and ol % 5 == 0:
                #     print('Gradient matching loss:', loss.item())
                if ol == outer_loop - 1:
                    # print('loss_reg:', loss_reg.item())
                    # print('Gradient matching loss:', loss.item())
                    break

                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                adj_syn_inner_norm = normalize_adj_tensor(adj_syn_inner, sparse=False)

                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    # print(loss_syn_inner.item())
                    optimizer_model.step()  # update gnn param

            loss_avg /= (data.nclass * outer_loop)
            if verbose and it % 10 == 0:
                print('Epoch {}, loss_avg: {}'.format(it, loss_avg))

            eval_epochs = [50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1600, 2000, 3000, 4000, 5000]
            # if it == 0:

            if it in eval_epochs:
                # if verbose and (it+1) % 50 == 0:
                data.adj_syn, data.feat_syn, data.labels_syn = adj_syn_inner.detach(), feat_syn_inner.detach(), labels_syn.detach()
                res = []
                for i in range(3):
                    res.append(self.test_with_val(verbose=verbose, setting=args.setting))

                res = np.array(res)
                current_val = res.mean()
                if verbose:
                    print('Val Accuracy and Std:',
                          repr([current_val, res.std()]))

                if current_val > best_val:
                    best_val = current_val
                    save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)

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
