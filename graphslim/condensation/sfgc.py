import os
from copy import deepcopy

import scipy

from graphslim.condensation.gcond_base import GCondBase
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.models import *
from graphslim.models.gntk import GNTK
from graphslim.models.reparam_module import ReparamModule
from graphslim.sparsification import *
from graphslim.utils import *
from tqdm import trange


class SFGC(GCondBase):
    def __init__(self, setting, data, args, **kwargs):
        super(SFGC, self).__init__(setting, data, args, **kwargs)

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args
        # =============stage 1 trajectory save and load==================#
        # can skip to save time
        assert args.teacher_epochs + 100 >= args.expert_epochs
        args.condense_model = 'GCN'
        args.init = 'kcenter'
        buf_dir = '../SFGC_Buffer/{}'.format(args.dataset)
        if not args.no_buff:
            args.condense_model = 'GCN'
            args.num_experts = 100
            if not os.path.exists(buf_dir):
                os.mkdir(buf_dir)

            if args.setting == 'ind':
                features, adj, labels = to_tensor(data.feat_train, data.adj_train, data.labels_train)
            else:
                features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full,
                                                  device=self.device)
            adj = normalize_adj_tensor(adj, sparse=True)
            device = args.device

            trajectories = []

            model = eval(args.condense_model)(features.shape[1], args.hidden, data.nclass, args).to(device)
            for it in trange(args.num_experts):

                model.initialize()

                model_parameters = list(model.parameters())

                optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr, weight_decay=args.weight_decay)

                timestamps = []

                timestamps.append([p.detach().cpu() for p in model.parameters()])

                for e in range(args.teacher_epochs):
                    model.train()
                    optimizer_model.zero_grad()
                    output = model.forward(features, adj)
                    if args.setting == 'ind':
                        loss_buffer = F.nll_loss(output, labels)
                    else:
                        loss_buffer = F.nll_loss(output[data.idx_train], labels[data.idx_train])
                    loss_buffer.backward()
                    optimizer_model.step()

                    if e % 10 == 0 and e > 1:
                        timestamps.append([p.detach().cpu() for p in model.parameters()])

                trajectories.append(timestamps)

                # need too many space to save
                if len(trajectories) == 10:
                    n = 0
                    while os.path.exists(os.path.join(buf_dir, "replay_buffer_{}.pt".format(n))):
                        n += 1
                    print("Saving {}".format(os.path.join(buf_dir, "replay_buffer_{}.pt".format(n))))
                    torch.save(trajectories, os.path.join(buf_dir, "replay_buffer_{}.pt".format(n)))
                    trajectories = []
        # =============stage 2 trajectory alignment and GCN evaluation==================#
        # kcenter select
        feat_init = self.init_feat()
        self.feat_syn.data.copy_(feat_init)
        self.labels_syn = to_tensor(label=data.labels_syn, device=self.device)
        # self.adj_syn_init = adj_init

        file_idx, expert_idx, expert_files = self.expert_load(buf_dir)

        # args.lr_student
        syn_lr = torch.tensor(args.lr_student).float()
        syn_lr = syn_lr.detach().to(self.device).requires_grad_(True)
        optimizer_lr = torch.optim.SGD([syn_lr], lr=1e-6, momentum=0.5)

        best_val = 0

        bar = trange(args.epochs)
        for it in bar:
            model = eval(args.condense_model)(data.feat_train.shape[1], args.hidden, data.nclass, args).to(self.device)

            model = ReparamModule(model)

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

            start = np.linspace(0, args.start_epoch, num=args.start_epoch // 10 + 1)
            start_epoch = int(np.random.choice(start, 1)[0])
            start_epoch = start_epoch // 10

            starting_params = expert_trajectory[start_epoch]

            target_params = expert_trajectory[start_epoch + args.expert_epochs // 10]

            target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params], 0)

            student_params = [
                torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

            starting_params = torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0)

            param_loss_list = []
            param_dist_list = []

            # if it == 0:
            #     feat_syn = self.feat_syn
            #     adj_syn_norm = normalize_adj_tensor(self.adj_syn_init, sparse=True)
            #     adj_syn_input = adj_syn_norm
            # else:
            feat_syn = self.feat_syn
            adj_syn = torch.eye(feat_syn.shape[0]).to(self.device)
            adj_syn_cal_norm = normalize_adj_tensor(adj_syn, sparse=False)
            adj_syn_input = adj_syn_cal_norm
            for step in range(args.syn_steps):
                forward_params = student_params[-1]
                output_syn = model.forward(feat_syn, adj_syn_input, flat_param=forward_params)
                loss_syn = F.nll_loss(output_syn, self.labels_syn)
                grad = torch.autograd.grad(loss_syn, student_params[-1], create_graph=True)[0]
                student_params.append(student_params[-1] - syn_lr * grad)

            param_loss = torch.tensor(0.0).to(self.device)
            param_dist = torch.tensor(0.0).to(self.device)

            param_loss += F.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist += F.mse_loss(starting_params, target_params, reduction="sum")
            param_loss_list.append(param_loss)
            param_dist_list.append(param_dist)

            param_loss /= num_params
            param_dist /= num_params

            param_loss /= param_dist
            total_loss = param_loss

            self.optimizer_feat.zero_grad()
            optimizer_lr.zero_grad()

            total_loss.backward()
            self.optimizer_feat.step()
            optimizer_lr.step()
            if torch.isnan(total_loss):
                break  # Break out of the loop if either is NaN
            bar.set_postfix_str(
                f"File ID = {file_idx} Total_Loss = {total_loss.item():.4f} Syn_Lr = {syn_lr.item():.4f}")
            if verbose and (it + 1) % 100 == 0:
                print('Epoch {}, loss_avg: {}'.format(it + 1, total_loss.item()))

            if it + 1 in args.checkpoints:
                data.adj_syn, data.feat_syn, data.labels_syn = torch.eye(
                    feat_syn.shape[0]), feat_syn.detach(), self.labels_syn.detach()
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
                    save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)

            for _ in student_params:
                del _

        return data

    def expert_load(self, expert_dir):
        '''
        randomly select one expert from expert files
        '''

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

        # print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])

        random.shuffle(buffer)
        self.buffer = buffer

        return file_idx, expert_idx, expert_files
