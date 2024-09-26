from tqdm import trange

from graphslim.condensation.gcond_base import GCondBase
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.utils import *
from graphslim.models import *
import torch.nn.functional as F


class GCDM(GCondBase):
    def __init__(self, setting, data, args, **kwargs):
        super(GCDM, self).__init__(setting, data, args, **kwargs)

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args
        self.feat_syn, labels_syn = to_tensor(self.feat_syn, label=data.labels_syn, device=self.device)
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train,
                                              device=self.device)

        # initialization the features
        feat_init = self.init()
        # self.reset_parameters()
        self.feat_syn.data.copy_(feat_init)

        adj = normalize_adj_tensor(adj, sparse=True)

        outer_loop, inner_loop = self.get_loops(args)
        model = eval(args.condense_model)(self.d, args.hidden,
                                          data.nclass, args).to(self.device)

        feat_syn = self.feat_syn
        adj_syn = torch.eye(feat_syn.shape[0], device=self.device)

        best_val = 0
        bar = trange(args.epochs)
        for it in bar:
            model.initialize()
            model_parameters = list(model.parameters())
            self.optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr)
            model.train()
            with torch.no_grad():
                emb_real,_ = model.forward(features,adj, output_layer_features=True)

            loss_avg = 0
            for ol in range(outer_loop):
                emb_syn, _ = model.forward(feat_syn,adj_syn, output_layer_features=True)
                loss_emb = 0
                # To parallelize
                for i in range(len(emb_syn)):
                    if i == args.nlayers-1:
                        break
                    for c in range(data.nclass):
                        coeff = self.num_class_dict[c] / self.nnodes_syn

                        st_id, ed_id = self.syn_class_indices[c]
                        num_syn_samples = emb_syn[i][st_id:ed_id].shape[0]
                        class_mask = (data.labels_train == c)
                        real_indices = class_mask.nonzero(as_tuple=False).squeeze()

                        num_real_samples = real_indices.shape[0]

                        selected_indices = real_indices[torch.randperm(num_real_samples)[:num_syn_samples]]

                        #emb_real_selected = emb_real[i][data.train_mask][selected_indices]
                        emb_real_class = emb_real[i][data.train_mask][class_mask]
                        emb_syn_selected = emb_syn[i][st_id:ed_id]

                        loss_emb += coeff * dist(torch.mean(emb_real_class,dim=0), torch.mean(emb_syn_selected,dim=0), method=args.dis_metric)

                loss_avg += loss_emb.item()

                self.optimizer_feat.zero_grad()
                loss_emb.backward()
                self.optimizer_feat.step()

                feat_syn_inner = feat_syn.detach()


                for _ in range(inner_loop):
                    props = model.forward(feat_syn_inner, adj_syn)
                    loss_inner = F.nll_loss(props, labels_syn)
                    self.optimizer_model.zero_grad()
                    loss_inner.backward()
                    self.optimizer_model.step()
                with torch.no_grad():
                    emb_real,_ = model.forward(features,adj, output_layer_features=True)
            loss_avg /= outer_loop
            bar.set_postfix({'loss': loss_avg})

            if it in args.checkpoints:
                self.adj_syn = adj_syn
                data.adj_syn, data.feat_syn, data.labels_syn = self.adj_syn, self.feat_syn.detach(), labels_syn.detach()
                best_val = self.intermediate_evaluation(best_val, loss_avg)

        return data
        
def dist(x, y, method='mse'):
    """Distance objectives
    """
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))
    return dist_

