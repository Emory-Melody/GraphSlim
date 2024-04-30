from graphslim.condensation.gcond_base import GCondBase
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.models import *
from graphslim.utils import *
from tqdm import trange


class GCondX(GCondBase):
    def __init__(self, setting, data, args, **kwargs):
        super(GCondX, self).__init__(setting, data, args, **kwargs)

    @verbose_time_memory
    def reduce(self, data, verbose=True):

        args = self.args
        feat_syn, pge, labels_syn = to_tensor(self.feat_syn, self.pge, label=data.labels_syn, device=self.device)
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train,
                                              device=self.device)
        syn_class_indices = self.syn_class_indices

        # initialization the features
        feat_sub, adj_sub = self.get_sub_adj_feat()
        self.feat_syn.data.copy_(feat_sub)
        self.adj_syn = torch.eye(feat_sub.shape[0], device=self.device)

        adj = normalize_adj_tensor(adj, sparse=True)

        outer_loop, inner_loop = self.get_loops(args)
        loss_avg = 0
        best_val = 0

        for it in range(args.epochs):
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
            # optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr)
            model.train()

            for ol in range(outer_loop):
                adj_syn_norm = self.adj_syn
                model = self.check_bn(model)
                loss = self.train_class(model, adj, features, labels, labels_syn, args)

                loss_avg += loss.item()

                self.optimizer_feat.zero_grad()
                loss.backward()
                self.optimizer_feat.step()

                # if args.normalize_features:
                #     feat_syn_inner_norm = F.normalize(feat_syn_inner, dim=0)
                # else:
                #     feat_syn_inner_norm = feat_syn_inner
                # for j in range(inner_loop):
                #     optimizer_model.zero_grad()
                #     output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                #     loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                #     loss_syn_inner.backward()
                #     # print(loss_syn_inner.item())
                #     optimizer_model.step()  # update gnn param

            loss_avg /= (data.nclass * outer_loop)
            if verbose and (it + 1) % 100 == 0:
                print('Epoch {}, loss_avg: {}'.format(it + 1, loss_avg))

            # eval_epochs = [400, 600, 800, 1000, 1200, 1600, 2000, 3000, 4000, 5000]
            # if it == 0:

            if it + 1 in args.checkpoints:
                data.adj_syn, data.feat_syn, data.labels_syn = adj_syn_norm.detach(), feat_syn.detach(), labels_syn.detach()
                res = []
                for i in range(3):
                    res.append(self.test_with_val(verbose=False, setting=args.setting))

                res = np.array(res)
                current_val = res.mean()
                if verbose:
                    print('Val Accuracy and Std:',
                          repr([current_val, res.std()]))

                if current_val > best_val:
                    best_val = current_val
                    save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args, best_val)

        return data
