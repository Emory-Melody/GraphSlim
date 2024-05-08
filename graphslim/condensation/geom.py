import os

import torch.nn as nn
from tqdm import trange
import copy

from graphslim.condensation.gcond_base import GCondBase
from graphslim.dataset.utils import save_reduced
from graphslim.condensation.utils import sort_training_nodes, sort_training_nodes_in, training_scheduler
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.sparsification import *
from graphslim.utils import *
from graphslim.models.reparam_module import ReparamModule
from graphslim.models import *


class GEOM(GCondBase):
    def __init__(self, setting, data, args, **kwargs):
        super(GEOM, self).__init__(setting, data, args, **kwargs)
        assert args.teacher_epochs + 100 >= args.expert_epochs
        args.condense_model = 'GCN'
        args.init = 'kcenter'

        # n = int(data.feat_train.shape[0] * args.reduction_rate)
        # d = data.feat_train.shape[1]
        # self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(self.device))
        #
        # if args.optimizer_con == 'Adam':
        #     self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        # elif args.optimizer_con == 'SGD':
        #     self.optimizer_feat = torch.optim.SGD([self.feat_syn], lr=args.lr_feat, momentum=0.9)
        self.buf_dir = '../../data/GEOM_Buffer/{}'.format(args.dataset)
        if not os.path.exists(self.buf_dir):
            os.makedirs(self.buf_dir)

        # print('adj_syn: {}, feat_syn: {}'.format((n, n), self.feat_syn.shape))

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args
        data = self.data

        if not args.no_buff:
            print("=================Begin buffer===============")
            self.buffer_cl(data)
            print("=================Finish buffer===============")

        graph_init = self.init_by_coreset()
        feat_init, adj_init, labels_init = to_tensor(graph_init.feat_syn, graph_init.adj_syn,
                                                     label=graph_init.labels_syn, device=self.device)
        self.feat_syn.data.copy_(feat_init)
        self.labels_syn = labels_init
        self.adj_syn_init = adj_init
        # features_tensor, adj_tensor, labels_tensor = to_tensor(data.feat_full, data.adj_full, labels=data.labels_full, device=self.device)

        file_idx, expert_idx, expert_files = self.expert_load()

        if args.soft_label == 1:
            model_4_soft = eval(args.condense_model)(data.feat_train.shape[1], args.hidden, data.nclass, args).to(
                self.device)

            model_4_soft = ReparamModule(model_4_soft)

            model_4_soft.eval()
            Temp_params = self.buffer[0][-1]
            Initialize_Labels_params = torch.cat([p.data.to(args.device).reshape(-1) for p in Temp_params], 0)
            # if args.distributed:
            #     Initialize_Labels_params = Initialize_Labels_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)

            # adj_syn_norm = utils.normalize_adj_tensor(self.adj_syn_init, sparse=True)
            # adj_syn_input = adj_syn_norm
            adj_syn = torch.eye(self.feat_syn.shape[0]).to(self.device)
            adj_syn_cal_norm = normalize_adj_tensor(adj_syn, sparse=is_sparse_tensor(adj_syn))
            adj_syn_input = adj_syn_cal_norm

            feat_4_soft, adj_4_soft = copy.deepcopy(self.feat_syn.detach()), copy.deepcopy(
                adj_syn_input.detach())
            label_soft = model_4_soft.forward(feat_4_soft, adj_4_soft, flat_param=Initialize_Labels_params)

            max_pred, pred_lab = torch.max(label_soft, dim=1)

            for i in range(labels_init.shape[0]):
                if pred_lab[i] != labels_init[i]:
                    label_soft[i][labels_init[i]] = max_pred[i]
                    # label_soft[i].fill_(0)
                    # label_soft[i][labels_init[i]] = 1

            self.labels_syn = copy.deepcopy(label_soft.detach()).to(args.device).requires_grad_(True)
            self.labels_syn.requires_grad = True
            self.labels_syn = self.labels_syn.to(args.device)

            acc = np.sum(np.equal(np.argmax(label_soft.cpu().data.numpy(), axis=-1), labels_init.cpu().data.numpy()))
            print('InitialAcc:{}'.format(acc / len(self.labels_syn)))

            self.optimizer_label = torch.optim.SGD([self.labels_syn], lr=args.lr_y, momentum=0.9)
            # -------------------------------------softlabel-------------------------------------------------------end-----------------------------------------------------------------#
        else:
            self.labels_syn = labels_init

        self.syn_lr = torch.tensor(args.lr_student).to(self.device)

        if args.optim_lr == 1:
            self.syn_lr = self.syn_lr.detach().to(self.device).requires_grad_(True)
            optimizer_lr = torch.optim.SGD([self.syn_lr], lr=1e-6, momentum=0.5)

        eval_it_pool = np.arange(0, args.epochs + 1, args.eval_interval).tolist()

        best_val = 0

        bar = trange(args.epochs + 1)
        for it in bar:
            model = eval(args.condense_model)(data.feat_train.shape[1], args.hidden, data.nclass, args).to(self.device)
            model_4_clom = eval(args.condense_model)(data.feat_train.shape[1], args.hidden, data.nclass, args).to(
                self.device)

            model = ReparamModule(model)
            model_4_clom = ReparamModule(model_4_clom)

            model.train()

            num_params = sum([np.prod(p.size()) for p in (model.parameters())])

            expert_trajectory = self.buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(self.buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                # print("loading file {}".format(expert_files[file_idx]))
                del self.buffer
                self.buffer = torch.load(expert_files[file_idx])
                random.shuffle(self.buffer)

            # expanding window
            Upper_Bound = args.max_start_epoch_s + it
            Upper_Bound = min(Upper_Bound, args.max_start_epoch)

            np.random.seed(it)
            start_epoch = np.random.randint(args.min_start_epoch, Upper_Bound)
            np.random.seed(args.seed_student)
            start_epoch = start_epoch // 10
            starting_params = expert_trajectory[start_epoch]

            # if args.interval_buffer == 1:
            target_params = expert_trajectory[start_epoch + args.expert_epochs // 10]
            target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params], 0)

            if args.beta:
                target_params_4_clom = expert_trajectory[-1]
                target_params_4_clom = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params_4_clom], 0)
                params_dict = dict(model_4_clom.named_parameters())
                for (name, param) in params_dict.items():
                    param.data.copy_(target_params_4_clom)
                model_4_clom.load_state_dict(params_dict)
                for param in model_4_clom.parameters():
                    param.requires_grad = False

            student_params = [
                torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

            starting_params = torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0)

            param_loss_list = []
            param_dist_list = []

            feat_syn = self.feat_syn
            adj_syn = torch.eye(feat_syn.shape[0]).to(self.device)
            adj_syn_cal_norm = normalize_adj_tensor(adj_syn, sparse=False)
            adj_syn_input = adj_syn_cal_norm

            # tag
            for step in range(args.syn_steps):
                forward_params = student_params[-1]
                output_syn = model.forward(feat_syn, adj_syn_input, flat_param=forward_params)

                if args.soft_label == 1:
                    loss_syn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(output_syn, self.labels_syn)
                else:
                    loss_syn = F.nll_loss(output_syn, self.labels_syn)

                grad = torch.autograd.grad(loss_syn, student_params[-1], create_graph=True)[0]

                student_params[-1] = student_params[-1] - self.syn_lr * grad

            param_loss = torch.tensor(0.0).to(self.device)
            param_dist = torch.tensor(0.0).to(self.device)

            param_loss += torch.norm(student_params[-1] - target_params, 2)
            param_dist += torch.norm(starting_params - target_params, 2)

            param_loss_list.append(param_loss)
            param_dist_list.append(param_dist)

            param_loss /= num_params
            param_dist /= num_params

            param_loss /= param_dist

            grand_loss = param_loss

            if args.beta == 0:
                total_loss = grand_loss
            else:
                output_clom = model_4_clom.forward(feat_syn, adj_syn_input,
                                                   flat_param=target_params_4_clom)
                if args.soft_label:
                    loss_clom = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(output_clom, self.labels_syn)
                else:
                    loss_clom = F.nll_loss(output_clom, self.labels_syn)
                total_loss = grand_loss + args.beta * loss_clom

            self.optimizer_feat.zero_grad()
            if args.soft_label:
                self.optimizer_label.zero_grad()
            if args.optim_lr:
                optimizer_lr.zero_grad()

            total_loss.backward()

            self.optimizer_feat.step()
            if args.soft_label:
                self.optimizer_label.step()
            if args.optim_lr:
                optimizer_lr.step()

            if torch.isnan(total_loss) or torch.isnan(grand_loss):
                break  # Break out of the loop if either is NaN
            bar.set_postfix_str(
                f"File ID = {file_idx} Total_Loss = {total_loss.item():.4f} Syn_Lr = {self.syn_lr.item():.4f}")

            if it in eval_it_pool and it > 0:
                feat_syn_save, adj_syn_save, label_syn_save = self.synset_save()
                data.adj_syn, data.feat_syn, data.labels_syn = adj_syn_save, feat_syn_save, label_syn_save
                best_val = self.intermediate_evaluation(best_val, total_loss.item())

            # if it % 1000 == 0 or it == args.ITER:
            #     feat_syn_save, adj_syn_save, label_syn_save = self.synset_save()
            #     torch.save(adj_syn_save,
            #                f'{args.log_dir}/adj_{args.dataset}_{args.reduction_rate}_{it}_{args.seed_student}_ours.pt')
            #     torch.save(feat_syn_save,
            #                f'{args.log_dir}/feat_{args.dataset}_{args.reduction_rate}_{it}_{args.seed_student}_ours.pt')
            #     torch.save(label_syn_save,
            #                f'{args.log_dir}/label_{args.dataset}_{args.reduction_rate}_{it}_{args.seed_student}_ours.pt')
            # for _ in student_params:
            #     del _

            # writer.add_scalar('grand_loss_curve', grand_loss.item(), it)
            # torch.cuda.empty_cache()

            # gc.collect()
        return data

    def buffer_cl(self, data):
        args = self.args

        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, data.labels_train,
                                              device=self.device)
            # features_sort, adj_sort, labels_sort = to_tensor(data.feat_full, data.adj_full, label=data.labels_full,
            #                                   device=self.device)
            # feat_val, adj_val, labels_val = to_tensor(data.feat_val, data.adj_val, label=data.labels_val,
            #                                                  device=self.device)
            # feat_test, adj_test, labels_test = to_tensor(data.feat_test, data.adj_test, label=data.labels_test,
            #                                                  device=self.device)
            # adj_sort = normalize_adj_tensor(adj_sort, sparse=is_sparse_tensor(adj))

        adj = normalize_adj_tensor(adj, sparse=is_sparse_tensor(adj))
        device = args.device

        trajectories = []
        model = eval(args.condense_model)(features.shape[1], args.hidden, data.nclass, args).to(device)

        adj_coo = adj.to_torch_sparse_coo_tensor()
        if args.setting == "trans":
            sorted_trainset = sort_training_nodes(data, adj_coo, labels)
        else:
            sorted_trainset = sort_training_nodes_in(data, adj_coo, labels)

        for it in trange(args.num_experts):
            # print(
            #     '======================== {} -th number of experts for {}-model_type=============================='.format(
            #         it, args.condense_model))

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

            if args.setting == 'ind' or args.dataset != 'citeseer':
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
                output = model.forward(features, adj)

                size = training_scheduler(args.lam, e, T, scheduler)

                training_subset = sorted_trainset[:int(size * sorted_trainset.shape[0])]

                loss_buffer = F.nll_loss(output[training_subset], labels[training_subset])

                # if args.setting == 'ind':
                #     acc_buffer = accuracy(output, labels)
                # else:
                #     acc_buffer = accuracy(output[data.idx_train], labels[data.idx_train])
                # print("Epochs: {} : Full graph train set results: loss= {:.4f}, accuracy= {:.4f} ".format(e,
                #                                                                                                  loss_buffer.item(),
                #                                                                                                  acc_buffer.item()))

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

                if e % 10 == 0 and e > 1:
                    timestamps.append([p.detach().cpu() for p in model.parameters()])
                    # p_current = timestamps[-1]
                    # p_0 = timestamps[0]
                    # target_params = torch.cat([p_c.data.reshape(-1) for p_c in p_current], 0)
                    # starting_params = torch.cat([p0.data.reshape(-1) for p0 in p_0], 0)
                    # param_dist1 = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
                    # print(
                    #     '==============================={}-th iter with length of {}-th tsp'.format(e, len(timestamps)))

            trajectories.append(timestamps)

            if len(trajectories) == 10:
                n = 0
                while os.path.exists(os.path.join(self.buf_dir, "replay_buffer_{}.pt".format(n))):
                    n += 1
                print("Saving {}".format(os.path.join(self.buf_dir, "replay_buffer_{}.pt".format(n))))
                torch.save(trajectories, os.path.join(self.buf_dir, "replay_buffer_{}.pt".format(n)))
                trajectories = []

    def expert_load(self):
        args = self.args
        expert_dir = self.buf_dir
        # logging.info("Expert Dir: {}".format(expert_dir))

        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)

        # if args.max_files is not None: # for ablation study
        #     expert_files = expert_files[:args.max_files]

        # print("loading file {}".format(expert_files[file_idx]))

        buffer = torch.load(expert_files[file_idx])

        # if args.max_experts is not None: # for ablation study
        #     buffer = buffer[:args.max_experts]

        random.shuffle(buffer)
        self.buffer = buffer

        return file_idx, expert_idx, expert_files

    def synset_save(self):
        args = self.args

        with torch.no_grad():
            feat_save = self.feat_syn
            eval_labs = self.labels_syn

        feat_syn_eval, label_syn_eval = copy.deepcopy(feat_save.detach()), copy.deepcopy(
            eval_labs.detach())  # avoid any unaware modification

        adj_syn_eval = torch.eye(feat_syn_eval.shape[0]).to(self.device)

        return feat_syn_eval, adj_syn_eval, label_syn_eval
