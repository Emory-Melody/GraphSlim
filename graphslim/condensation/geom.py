from graphslim.condensation.utils import sort_training_nodes, training_scheduler
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.utils import *


class GEOM:
    def __init__(self, setting, data, args, **kwargs):
        self.data = data
        self.args = args
        self.setting = setting
        self.device = args.device

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        self.buffer_cl(data)

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
                    logging.info('NOTE! Decaying lr to :{}'.format(lr))
                    if args.optim == 'SGD':
                        optimizer_model = torch.optim.SGD(model_parameters, lr=lr, momentum=args.mom_teacher,
                                                          weight_decay=args.wd_teacher)
                    elif args.optim == 'Adam':
                        optimizer_model = torch.optim.Adam(model_parameters, lr=lr,
                                                           weight_decay=args.wd_teacher)

                    optimizer_model.zero_grad()

                if e % 20 == 0:
                    logging.info("Epochs: {} : Train set training:, loss= {:.4f}".format(e, loss_buffer.item()))
                    model.eval()
                    labels_val = torch.LongTensor(data.labels_val).to(device)
                    labels_test = torch.LongTensor(data.labels_test).to(device)

                    # Full graph
                    _, output = model.predict(data.feat_full, data.adj_full)
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    loss_test = F.nll_loss(output[data.idx_test], labels_test)

                    acc_val = utils.accuracy(output[data.idx_val], labels_val)
                    acc_test = utils.accuracy(output[data.idx_test], labels_test)

                    writer.add_scalar('val_set_loss_curve', loss_val.item(), e)
                    writer.add_scalar('val_set_acc_curve', acc_val.item(), e)

                    writer.add_scalar('test_set_loss_curve', loss_test.item(), e)
                    writer.add_scalar('test_set_acc_curve', acc_test.item(), e)

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
                    writer.add_scalar('param_change', param_dist1.item(), e)
                    logging.info(
                        '==============================={}-th iter with length of {}-th tsp'.format(e, len(timestamps)))

            logging.info("Valid set best results: accuracy= {:.4f}".format(best_val_acc.item()))
            logging.info(
                "Test set best results: accuracy= {:.4f} within best iteration = {}".format(best_test_acc.item(),
                                                                                            best_it))

            trajectories.append(timestamps)

            if len(trajectories) == args.traj_save_interval:
                n = 0
                while os.path.exists(os.path.join(log_dir, "replay_buffer_{}.pt".format(n))):
                    n += 1
                logging.info("Saving {}".format(os.path.join(log_dir, "replay_buffer_{}.pt".format(n))))
                torch.save(trajectories, os.path.join(log_dir, "replay_buffer_{}.pt".format(n)))
                trajectories = []
