from tqdm import trange

from graphslim.condensation.gcond_base import GCondBase
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.utils import *
from graphslim.models import *


class GCDM(GCondBase):
    def __init__(self, setting, data, args, **kwargs):
        super(GCDM, self).__init__(setting, data, args, **kwargs)

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args
        pge = self.pge
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

        feat_syn, pge= self.feat_syn, self.pge

        best_val = 0
        for it in range(args.epochs):
            model.initialize()
            model_parameters = list(model.parameters())
            self.optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr)
            model.train()
            with torch.no_grad():
                mean_emb_real=self.conv_emb(model,features,adj,labels,setting=args.setting,mask=data.train_mask)

            loss_avg = 0
            for ol in range(outer_loop):
                for param in model.parameters():
                    param.requires_grad_(False)

                adj_syn = pge(feat_syn)
                mean_emb_syn=self.conv_emb(model,feat_syn,adj_syn,labels_syn,setting='ind')
                loss_emb=torch.sum(torch.mean((mean_emb_syn-mean_emb_real)**2,dim=1))

                loss_avg += loss_emb.item()

                self.optimizer_pge.zero_grad()
                self.optimizer_feat.zero_grad()

                loss_emb.backward()
                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                if ol == outer_loop - 1:
                    break

                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                adj_syn_inner_norm = normalize_adj_tensor(
                    adj_syn_inner, sparse=False
                )

                for param in model.parameters():
                    param.requires_grad_(True)


                for _ in range(inner_loop):
                    mean_emb_syn=self.conv_emb(model,feat_syn_inner,adj_syn_inner_norm,labels_syn,setting='ind')
                    loss_inner=torch.sum(torch.mean((mean_emb_syn-mean_emb_real)**2,dim=1))
                    self.optimizer_model.zero_grad()
                    loss_inner.backward()
                    self.optimizer_model.step()
                    with torch.no_grad():
                        mean_emb_real=self.conv_emb(model,features,adj,labels,setting=args.setting,mask=data.train_mask)
            loss_avg /= outer_loop

            if it in args.checkpoints:
                self.adj_syn = adj_syn_inner_norm
                data.adj_syn, data.feat_syn, data.labels_syn = self.adj_syn, self.feat_syn.detach(), labels_syn.detach()
                best_val = self.intermediate_evaluation(best_val, loss_avg)

        return data
    def conv_emb(self,model,features,adj,labels,setting='trans',mask=None):
        embedding_real, _ = model.forward(features, adj, output_layer_features=True)
        if setting=='trans':
            embedding_real = embedding_real[mask]
            y = labels[mask]
        else:
            y = labels
        mean_emb = torch.stack([torch.mean(embedding_real[y == cls], 0) for cls in torch.unique(y)])
        return mean_emb
        
