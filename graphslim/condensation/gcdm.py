from tqdm import trange

from graphslim.condensation.gcond_base import GCondBase
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.utils import *
from graphslim.models import *


class GCDM(GCondBase):
    """
    "Graph Condensation for Graph Neural Networks" https://cse.msu.edu/~jinwei2/files/GCond.pdf
    """
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
        loss_avg = 0
        best_val = 0
        model = eval(args.condense_model)(self.d, args.hidden,
                                          data.nclass, args).to(self.device)

        feat_syn, pge= self.feat_syn, self.pge

        adj_syn = pge.inference(feat_syn)
        for it in range(args.epochs):
            model.initialize()
            model_parameters = list(model.parameters())
            self.optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr)
            model.train()

            labels_unique = self.num_class_dict.keys()
            loss_avg = 0

            for ol in range(outer_loop):
                with torch.no_grad():
                    embedding_real, _ = model.forward(features, adj, output_layer_features=True)
                    if args.setting=='trans':
                        embedding_real = embedding_real[data.idx_train]
                        train_labels = labels[data.idx_train]
                    else:
                        train_labels = labels
                    mean_emb_real = torch.zeros(
                        (len(labels_unique), embedding_real.size(1)),
                        device=embedding_real.device,
                    )
                    for i, label in enumerate(labels_unique):
                        label_mask = train_labels == label
                        mean_emb_real[i] = torch.mean(embedding_real[label_mask], dim=0)

                adj_syn = pge(feat_syn)
                embedding_syn, _ = model.forward(feat_syn, adj_syn, output_layer_features=True)
                mean_emb_syn = torch.zeros(
                    (len(labels_unique), embedding_syn.size(1)),
                    device=embedding_syn.device,
                )
                for i, label in enumerate(labels_unique):
                    label_mask = labels_syn == label
                    mean_emb_syn[i] = torch.mean(embedding_syn[label_mask], dim=0)

                # loss_emb = torch.sum(
                #     torch.mean((mean_emb_syn - mean_emb_real) ** 2, dim=1)
                # )
                loss_emb = torch.mean((mean_emb_syn - mean_emb_real) ** 2).sum()
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

                for _ in range(inner_loop):
                    self.optimizer_model.zero_grad()
                    embedding_syn, _ = model.forward(
                        feat_syn_inner, adj_syn_inner_norm, output_layer_features=True
                    )
                    mean_emb_syn = torch.zeros(
                        (len(labels_unique), embedding_syn.size(1)),
                        device=embedding_syn.device,
                    )
                    for i, label in enumerate(labels_unique):
                        label_mask = labels_syn == label
                        mean_emb_syn[i] = torch.mean(embedding_syn[label_mask], dim=0)
                    loss_syn_inner = torch.mean(
                        (mean_emb_syn - mean_emb_real) ** 2
                    ).sum()
                    # loss_syn_inner = torch.sum(
                    #     torch.mean((mean_emb_syn - mean_emb_real) ** 2, dim=1)
                    # )
                    loss_syn_inner.backward()
                    self.optimizer_model.step()
                    with torch.no_grad():
                        embedding_real, _ = model.forward(
                            features, adj, output_layer_features=True
                        )
                        if args.setting =='trans':
                            embedding_real = embedding_real[data.idx_train]
                        mean_emb_real = torch.zeros(
                            (len(labels_unique), embedding_real.size(1)),
                            device=embedding_real.device,
                        )
                        for i, label in enumerate(labels_unique):
                            label_mask = train_labels == label
                            mean_emb_real[i] = torch.mean(
                                embedding_real[label_mask], dim=0
                            )
                # self.feat_syn = feat_syn
                # self.pge = pge
            loss_avg /= data.nclass * outer_loop

            if it in args.checkpoints:
                self.adj_syn = adj_syn_inner_norm
                data.adj_syn, data.feat_syn, data.labels_syn = self.adj_syn, self.feat_syn.detach(), labels_syn.detach()
                best_val = self.intermediate_evaluation(best_val, loss_avg)

        return data
