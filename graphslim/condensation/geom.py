import os

import torch.nn as nn

from graphslim.condensation.utils import sort_training_nodes, training_scheduler
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.utils import *


class GEOM:
    def __init__(self, setting, data, args, **kwargs):
        self.data = data
        self.args = args
        self.setting = setting
        self.device = args.device

        n = int(data.feat_train.shape[0] * args.reduction_rate)
        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(self.device))

        if args.optimizer_con == 'Adam':
            self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        elif args.optimizer_con == 'SGD':
            self.optimizer_feat = torch.optim.SGD([self.feat_syn], lr=args.lr_feat, momentum=0.9)

        print('adj_syn: {}, feat_syn: {}'.format((n, n), self.feat_syn.shape))

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args
        data = self.data
        if not os.path.exists(f"./output/geom_buffer/{args.dataset}"):
            self.buffer_cl(data)

        features, adj, labels = data.feat_full, data.adj_full, data.labels_full
        features_tensor, adj_tensor, labels_tensor = to_tensor(features, adj, label=labels, device=self.device)

        feat_init, adj_init, labels_init = self.get_coreset_init()
        feat_init, adj_init, labels_init = to_tensor(feat_init, adj_init, label=labels_init, device=self.device)

        adj_tensor_norm = normalize_adj_tensor(adj_tensor, sparse=is_sparse_tensor(adj))

        self.feat_syn.data.copy_(feat_init)
        self.labels_syn = labels_init
        self.adj_syn_init = adj_init




    def buffer_cl(self, data):
        args = self.args
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train,
                                              device=self.device)

        adj = normalize_adj_tensor(adj, sparse=is_sparse_tensor(adj))

        trajectories = []

        model_type = args.buffer_model_type
        sorted_trainset = sort_training_nodes(data, adj, labels)

        for it in range(0, args.num_experts):
            print(
                '======================== {} -th number of experts for {}-model_type=============================='.format(
                    it, model_type))

            model_class = eval(model_type)

            model = model_class(nfeat=features.shape[1], nhid=args.teacher_hidden, dropout=args.teacher_dropout,
                                nlayers=args.teacher_nlayers,
                                nclass=data.nclass, device=self.device).to(self.device)

            model.initialize()

            model_parameters = list(model.parameters())

            if args.optim == 'Adam':
                optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_teacher, weight_decay=args.wd_teacher)
            elif args.optim == 'SGD':
                optimizer_model = torch.optim.SGD(model_parameters, lr=args.lr_teacher, momentum=args.mom_teacher,
                                                  weight_decay=args.wd_teacher)

            timestamps = []

            timestamps.append([p.detach().cpu() for p in model.parameters()])

            best_val_acc = best_test_acc = best_it = 0

            if args.dataset != 'citeseer':
                lr_schedule = [args.teacher_epochs // 2 + 1]
            else:
                lr_schedule = [600]

            lr = args.lr_teacher
            lam = float(args.lam)
            T = float(args.T)
            args.lam = lam
            args.T = T
            scheduler = args.scheduler

            for e in range(args.teacher_epochs + 1):
                model.train()
                optimizer_model.zero_grad()
                _, output = model.forward(features, adj)

                size = training_scheduler(args.lam, e, T, scheduler)

                training_subset = sorted_trainset[:int(size * sorted_trainset.shape[0])]

                loss_buffer = F.nll_loss(output[training_subset], labels[training_subset])

                acc_buffer = accuracy(output[data.idx_train], labels[data.idx_train])

                print("Epochs: {} : Full graph train set results: loss= {:.4f}, accuracy= {:.4f} ".format(e,
                                                                                                          loss_buffer.item(),
                                                                                                          acc_buffer.item()))
                loss_buffer.backward()
                optimizer_model.step()

                if e in lr_schedule and args.decay:
                    lr = lr * args.decay_factor
                    # logging.info('NOTE! Decaying lr to :{}'.format(lr))
                    if args.optim == 'SGD':
                        optimizer_model = torch.optim.SGD(model_parameters, lr=lr, momentum=args.mom_teacher,
                                                          weight_decay=args.wd_teacher)
                    elif args.optim == 'Adam':
                        optimizer_model = torch.optim.Adam(model_parameters, lr=lr,
                                                           weight_decay=args.wd_teacher)

                    optimizer_model.zero_grad()

                if e % 20 == 0:
                    print("Epochs: {} : Train set training:, loss= {:.4f}".format(e, loss_buffer.item()))
                    model.eval()
                    labels_val = torch.LongTensor(data.labels_val).to(self.device)
                    labels_test = torch.LongTensor(data.labels_test).to(self.device)

                    # Full graph
                    _, output = model.predict(data.feat_full, data.adj_full)
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    loss_test = F.nll_loss(output[data.idx_test], labels_test)

                    acc_val = accuracy(output[data.idx_val], labels_val)
                    acc_test = accuracy(output[data.idx_test], labels_test)

                    if acc_val > best_val_acc:
                        best_val_acc = acc_val
                        best_test_acc = acc_test
                        best_it = e

                if e % args.param_save_interval == 0 and e > 1:
                    timestamps.append([p.detach().cpu() for p in model.parameters()])
                    p_current = timestamps[-1]
                    p_0 = timestamps[0]
                    target_params = torch.cat([p_c.data.reshape(-1) for p_c in p_current], 0)
                    starting_params = torch.cat([p0.data.reshape(-1) for p0 in p_0], 0)
                    param_dist1 = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
                    print(
                        '==============================={}-th iter with length of {}-th tsp'.format(e, len(timestamps)))

            print("Valid set best results: accuracy= {:.4f}".format(best_val_acc.item()))
            print(
                "Test set best results: accuracy= {:.4f} within best iteration = {}".format(best_test_acc.item(),
                                                                                            best_it))

            trajectories.append(timestamps)

            log_dir = f"./output/geom_buffer/{args.dataset}"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            if len(trajectories) == args.traj_save_interval:
                n = 0
                while os.path.exists(os.path.join(log_dir, "replay_buffer_{}.pt".format(n))):
                    n += 1
                print("Saving {}".format(os.path.join(log_dir, "replay_buffer_{}.pt".format(n))))
                torch.save(trajectories, os.path.join(log_dir, "replay_buffer_{}.pt".format(n)))
                trajectories = []

    def get_coreset_init(self, valid_result=0):
        args = self.args

        save_path = 'dataset/output'
        adj_syn = torch.load(
            f'{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.init_coreset_method}_{args.seed}_{valid_result}.pt')
        feat_syn = torch.load(
            f'{save_path}/feat_{args.dataset}_{args.reduction_rate}_{args.init_coreset_method}_{args.seed}_{valid_result}.pt')
        labels_syn = torch.load(
            f'{save_path}/label_{args.dataset}_{args.reduction_rate}_{args.init_coreset_method}_{args.seed}_{valid_result}.pt')
        if args.verbose:
            print("Loaded reduced data")

        if is_sparse_tensor(adj_syn):
            adj_syn = adj_syn.to_dense()

        return feat_syn, adj_syn, labels_syn
