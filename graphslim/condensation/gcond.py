from tqdm import trange

from graphslim.condensation.gcond_base import GCondBase
from graphslim.dataset.utils import save_reduced
from graphslim.models import *
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.utils import *


class GCond(GCondBase):

    def __init__(self, setting, data, args, **kwargs):
        super(GCond, self).__init__(setting, data, args, **kwargs)

    @verbose_time_memory
    def reduce(self, data, verbose=True):

        args = self.args
        feat_syn, pge, labels_syn = to_tensor(self.feat_syn, self.pge, label=data.labels_syn, device=self.device)
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train,
                                              device=self.device)

        # initialization the features
        feat_sub, adj_sub = self.get_sub_adj_feat()
        self.feat_syn.data.copy_(feat_sub)

        adj = normalize_adj_tensor(adj, sparse=True)

        outer_loop, inner_loop = self.get_loops(args)
        loss_avg = 0
        best_val = 0

        for it in trange(args.epochs):
            # seed_everything(args.seed + it)
            if args.dataset in ['ogbn-arxiv', 'flickr']:
                model = SGCRich(nfeat=feat_syn.shape[1], nhid=args.hidden,
                                dropout=0, with_bn=False,
                                weight_decay=0, nlayers=args.nlayers,
                                nclass=data.nclass, ntrans=2,
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
                adj_syn = pge(self.feat_syn)
                self.adj_syn = normalize_adj_tensor(adj_syn, sparse=False)
                model = self.check_bn(model)
                loss = self.train_class(model, adj, features, labels, labels_syn, args)
                loss_avg += loss.item()

                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()

                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                feat_syn_inner = self.feat_syn.detach()
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
            if verbose and (it + 1) % 100 == 0:
                print('Epoch {}, loss_avg: {}'.format(it + 1, loss_avg))

            # eval_epochs = [400, 600, 800, 1000, 1200, 1600, 2000, 3000, 4000, 5000]
            # if it == 0:

            if it + 1 in args.checkpoints:
                # if verbose and (it+1) % 50 == 0:
                data.adj_syn, data.feat_syn, data.labels_syn = adj_syn_inner.detach(), self.feat_syn.detach(), labels_syn.detach()
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
                    save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args, best_val)

        return data
