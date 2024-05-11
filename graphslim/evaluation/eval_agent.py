import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from sklearn.model_selection import ParameterGrid

from torch_geometric.utils import dense_to_sparse
from graphslim.dataset import *
from graphslim.models import *
from torch_sparse import SparseTensor
from graphslim.dataset.convertor import ei2csr
from graphslim.utils import accuracy, seed_everything, is_sparse_tensor, is_identity


class Evaluator:

    def __init__(self, args, **kwargs):
        # self.data = data
        self.args = args
        self.device = args.device

        self.reset_parameters()
        # print('adj_param:', self.adj_param.shape, 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        pass
        # self.adj_param.data.copy_(torch.randn(self.adj_param.size()))
        # self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    #
    def sparsify(self, model_type, adj_syn, verbose=True):
        args = self.args
        threshold = 0
        if model_type == 'MLP':
            adj_syn = adj_syn - adj_syn
            torch.diagonal(adj_syn).fill_(1)
        elif model_type == 'GAT':
            if args.method in ['gcond', 'doscond']:
                if args.dataset in ['cora', 'citeseer']:
                    threshold = 0.5  # Make the graph sparser as GAT does not work well on dense graph
                else:
                    threshold = 0.01
            elif args.method in ['msgc']:
                threshold = args.threshold
            else:
                threshold = 0
        else:
            if args.method in ['gcond', 'doscond']:
                threshold = args.threshold
            elif args.method in ['msgc']:
                threshold = 0
            else:
                threshold = 0
        if verbose and args.method not in ['gcondx', 'doscondx', 'sfgc', 'geom']:
            print('Sum:', adj_syn.sum().item(), (adj_syn.sum() / adj_syn.numel()))
            print('Sparsity:', adj_syn.nonzero().shape[0] / adj_syn.numel())
        if threshold > 0:
            adj_syn[adj_syn < threshold] = 0
            if verbose:
                print('Sparsity after truncating:', adj_syn.nonzero().shape[0] / adj_syn.numel())
            # else:
            #     print("structure free methods do not need to truncate the adjacency matrix")
        if model_type == 'GAT':
            # GATconv only supports sparse tensor
            return adj_syn.to_sparse_coo()
        else:
            return adj_syn

    def get_syn_data(self, model_type, verbose=False):

        args = self.args
        adj_syn, feat_syn, labels_syn = load_reduced(args, args.valid_result)
        if is_sparse_tensor(adj_syn):
            adj_syn = adj_syn.to_dense()
        elif isinstance(adj_syn, torch.sparse.FloatTensor):
            adj_syn = adj_syn.to_dense()
        else:
            adj_syn = adj_syn
        adj_syn = self.sparsify(model_type, adj_syn, verbose=verbose)

        return feat_syn, adj_syn, labels_syn

    def grid_search(self, data, model_type, param_grid):
        args = self.args
        args.eval_epochs = 100
        best_result = None
        best_params = None
        for params in ParameterGrid(param_grid):
            for key, value in params.items():
                setattr(args, key, value)

            res = []
            for i in range(args.run_eval):
                seed_everything(args.seed + i)
                res.append(self.test(data, model_type=model_type, verbose=False, reduced=True, mode='cross'))
            res = np.array(res)
            res_mean, res_std = res.mean(), res.std()
            if args.verbose:
                print(f'{model_type} Test with params {params}: {100 * res_mean:.2f} +/- {100 * res_std:.2f}')

            if best_result is None or res_mean > best_result:
                best_result = res_mean
                best_params = params

        print(f'Best {model_type} Result: {100 * best_result:.2f} +/- {100 * res_std:.2f} with params {best_params}')

    def train_cross(self, data):
        args = self.args
        args.valid_result = 0
        gs_params = {
            'MLP': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                    'dropout': [0.0, 0.5]},
            'GCN': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                    'dropout': [0.0, 0.5]},
            'SGC': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                    'dropout': [0.0, 0.5], 'ntrans': [1, 2]},
            'APPNP': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                      'dropout': [0.0, 0.5], 'ntrans': [1, 2], 'alpha': [0.1, 0.2]},
            'GraphSage': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                          'dropout': [0.0, 0.5]},
            'Cheby': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                      'dropout': [0.0, 0.5]},
            'GAT': {'hidden': [64, 128, 256], 'lr': [0.05, 0.01, 0.001, 0.005], 'weight_decay': [0, 5e-4],
                    'dropout': [0.0, 0.5, 0.7]}
        }
        for model_type in gs_params.keys():
            data.feat_syn, data.adj_syn, data.labels_syn = self.get_syn_data(model_type=model_type,
                                                                             verbose=args.verbose)
            print(f'Starting Grid Search for {model_type}')
            self.grid_search(data, model_type, gs_params[model_type])

    def test(self, data, model_type, verbose=True, reduced=True, mode='eval'):
        args = self.args

        if verbose:
            print('======= testing %s' % model_type)

        if model_type == 'MLP':
            adj_test = ei2csr(torch.arange(len(data.feat_test), dtype=torch.long).repeat(2, 1), len(data.feat_test))
            adj_full = ei2csr(torch.arange(len(data.feat_full), dtype=torch.long).repeat(2, 1), len(data.feat_full))
            model_type = 'GCN'
        else:
            adj_test = data.adj_test
            adj_full = data.adj_full

        # assert not (args.method not in ['msgc'] and model_type == 'GAT')
        model = eval(model_type)(data.feat_syn.shape[1], args.hidden, data.nclass, args, mode=mode).to(
            self.device)
        model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True, verbose=verbose,
                           setting=args.setting,
                           reduced=reduced)

        model.eval()
        labels_test = data.labels_test.long().to(args.device)

        res = []
        if args.setting == 'ind':
            output = model.predict(data.feat_test, adj_test)
            loss_test = F.nll_loss(output, labels_test)
            acc_test = accuracy(output, labels_test)
            res.append(acc_test.item())
            if verbose:
                print("Test set results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

        else:
            # Full graph
            output = model.predict(data.feat_full, adj_full)
            loss_test = F.nll_loss(output[data.idx_test], labels_test)
            acc_test = accuracy(output[data.idx_test], labels_test)
            res.append(acc_test.item())
            if verbose:
                print("Test full set results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

        return res[0]

    # def train_cross(self, data):
    #     args = self.args
    #     args.valid_result = 0
    #     for model_type in ['GAT']:  #
    #         data.feat_syn, data.adj_syn, data.labels_syn = self.get_syn_data(model_type=model_type,
    #                                                                          verbose=args.verbose)
    #         if args.verbose:
    #             run_evaluation = trange(args.run_evaluation)
    #         else:
    #             run_evaluation = range(args.run_evaluation)
    #         res = []
    #         for i in run_evaluation:
    #             seed_everything(args.seed + i)
    #             res.append(self.test(data, model_type=model_type, verbose=False, reduced=True, mode='eval'))
    #         res = np.array(res)
    #         res_mean, res_std = res.mean(), res.std()
    #         print(f'{model_type} Test Mean Result: {100 * res_mean:.2f} +/- {100 * res_std:.2f}')

    def evaluate(self, data, model_type, verbose=True, reduced=True):
        args = self.args

        data.feat_syn, data.adj_syn, data.labels_syn = self.get_syn_data(model_type=model_type, verbose=args.verbose)

        if verbose:
            print(f'evaluate reduced data by {model_type}')
            run_evaluation = trange(args.run_eval)
        else:
            run_evaluation = range(args.run_eval)

        res = []
        for i in run_evaluation:
            seed_everything(args.seed + i)
            best_val_acc = self.test(data, model_type=model_type, verbose=False, reduced=reduced, mode='eval')
            res.append(best_val_acc)
            if verbose:
                run_evaluation.set_postfix(best_val_acc=best_val_acc)
        res = np.array(res)

        if verbose:
            print(f'Test Mean Accuracy: {100 * res.mean():.2f} +/- {100 * res.std():.2f}')
        return res.mean(), res.std()

    def nas_evaluate(self, data, model_type, verbose=True, reduced=None):
        args = self.args
        res = []
        data.feat_syn, data.adj_syn, data.labels_syn = self.get_syn_data(model_type=model_type, verbose=args.verbose)
        if verbose:
            run_evaluation = trange(args.run_evaluation)
        else:
            run_evaluation = range(args.run_evaluation)
        for i in run_evaluation:
            model = eval(model_type)(data.feat_syn.shape[1], args.hidden, data.nclass, args, mode='eval').to(
                self.device)
            best_acc_val = model.fit_with_val(data,
                                              train_iters=args.eval_epochs,
                                              normadj=True,
                                              verbose=verbose,
                                              setting=args.setting,
                                              reduced=reduced)
            res.append(best_acc_val.item())
            if verbose:
                run_evaluation.set_postfix(best_acc_val=best_acc_val.item())
        res = np.array(res)

        if verbose:
            print(f'Test Mean Accuracy: {100 * res.mean():.2f} +/- {100 * res.std():.2f}')
        return res.mean(), res.std()
