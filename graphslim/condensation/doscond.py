from tqdm import trange

from graphslim.condensation.gcond_base import GCondBase
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.models import *
from graphslim.utils import *


class DosCond(GCondBase):
    def __init__(self, setting, data, args, **kwargs):
        super(DosCond, self).__init__(setting, data, args, **kwargs)

    @verbose_time_memory
    def reduce(self, data, verbose=True):

        args = self.args
        pge = self.pge
        feat_syn, labels_syn = to_tensor(self.feat_syn, label=data.labels_syn, device=self.device)
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train,
                                              device=self.device)

        # initialization the features
        feat_sub, adj_sub = self.get_sub_adj_feat()
        self.feat_syn.data.copy_(feat_sub)
        adj = normalize_adj_tensor(adj, sparse=is_sparse_tensor(adj))
        # self.sparsity = self.adj_syn.mean().item()
        # self.adj_syn.data.copy_(self.adj_syn * 10 - 5)  # max:5; min:-5


        outer_loop, inner_loop = self.get_loops(args)
        loss_avg = 0
        best_val = 0

        # seed_everything(args.seed + it)
        if args.dataset in ['ogbn-arxiv']:
            model = SGCRich(nfeat=feat_syn.shape[1], nhid=args.hidden,
                            dropout=0.0, with_bn=False,
                            weight_decay=0e-4, nlayers=args.nlayers,
                            nclass=data.nclass,
                            device=self.device).to(self.device)
        else:
            model = GCN(nfeat=feat_syn.shape[1], nhid=args.hidden, weight_decay=0,
                        nclass=data.nclass, dropout=0, nlayers=args.nlayers,
                        device=self.device).to(self.device)
        for it in trange(args.epochs):

            model.initialize()
            # model_parameters = list(model.parameters())
            # optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr)
            model.train()

            for ol in range(outer_loop):
                adj_syn = pge(self.feat_syn)
                self.adj_syn = normalize_adj_tensor(adj_syn, sparse=False)
                # feat_syn_norm = feat_syn
                model = self.check_bn(model)

                loss = self.train_class(model, adj, features, labels, labels_syn, args)

                loss_avg += loss.item()
                # if args.alpha > 0:
                #     loss_reg = args.alpha * regularization(adj_syn, tensor2onehot(labels_syn))
                # else:
                # loss_reg = torch.tensor(0)

                # loss = loss + loss_reg

                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()
                self.optimizer_pge.step()
                self.optimizer_feat.step()

            loss_avg /= (data.nclass * outer_loop)
            if verbose and (it + 1) % 100 == 0:
                print('Epoch {}, loss_avg: {}'.format(it + 1, loss_avg))

            # eval_epochs = [400, 600, 800, 1000, 1200, 1600, 2000, 3000, 4000, 5000]
            # eval_epochs = [400, 600, 1000]
            # if it == 0:

            if it + 1 in args.checkpoints:
                # if verbose and (it+1) % 50 == 0:
                self.adj_syn = pge.inference(self.feat_syn.detach())
                data.adj_syn, data.feat_syn, data.labels_syn = self.adj_syn.detach(), self.feat_syn.detach(), labels_syn.detach()
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
