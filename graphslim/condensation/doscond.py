from graphslim.condensation.gcond_base import GCondBase
from graphslim.condensation.utils import match_loss
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
        feat_syn, pge, labels_syn = to_tensor(self.feat_syn, self.pge, label=data.labels_syn, device=self.device)
        features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)

        syn_class_indices = self.syn_class_indices

        # initialization the features
        feat_sub, adj_sub = self.get_sub_adj_feat()
        self.feat_syn.data.copy_(feat_sub)
        # self.sparsity = self.adj_syn.mean().item()
        # self.adj_syn.data.copy_(self.adj_syn * 10 - 5)  # max:5; min:-5

        adj = normalize_adj_tensor(adj, sparse=True)

        outer_loop, inner_loop = self.get_loops(args)
        loss_avg = 0
        best_val = 0

        for it in range(args.epochs):
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

            model.initialize()
            model_parameters = list(model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr)
            model.train()

            for ol in range(outer_loop):
                adj_syn = pge(self.feat_syn)
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
                # if args.alpha > 0:
                #     loss_reg = args.alpha * regularization(adj_syn, tensor2onehot(labels_syn))
                # else:
                # loss_reg = torch.tensor(0)

                # loss = loss + loss_reg

                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()
                self.optimizer_feat.step()
                self.optimizer_pge.step()

                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)

            loss_avg /= (data.nclass * outer_loop)
            if verbose and (it + 1) % 100 == 0:
                print('Epoch {}, loss_avg: {}'.format(it + 1, loss_avg))

            # eval_epochs = [400, 600, 800, 1000, 1200, 1600, 2000, 3000, 4000, 5000]
            # eval_epochs = [400, 600, 1000]
            # if it == 0:

            if it + 1 in args.checkpoints:
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
