import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange, tqdm
from sklearn.model_selection import ParameterGrid

from torch_geometric.utils import dense_to_sparse
from graphslim.dataset import *
from graphslim.evaluation.utils import sparsify
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

    def get_syn_data(self, model_type, verbose=False):

        args = self.args
        adj_syn, feat_syn, labels_syn = load_reduced(args)

        if is_sparse_tensor(adj_syn):
            adj_syn = adj_syn.to_dense()
        elif isinstance(adj_syn, torch.sparse.FloatTensor):
            adj_syn = adj_syn.to_dense()
        else:
            adj_syn = adj_syn
        adj_syn = sparsify(model_type, adj_syn, args, verbose=verbose)
        return feat_syn, adj_syn, labels_syn

    def grid_search(self, data, model_type, param_grid, reduced=True):
        args = self.args
        best_result = None
        best_params = None
        for params in tqdm(ParameterGrid(param_grid)):
            for key, value in params.items():
                setattr(args, key, value)

            res = []
            for i in range(args.run_eval):
                seed_everything(i)
                res.append(self.test(data, model_type=model_type, verbose=False, reduced=reduced, mode='cross'))
                torch.cuda.empty_cache()
            res = np.array(res)
            res_mean, res_std = res.mean(), res.std()
            if args.verbose:
                print(f'{model_type} Test with params {params}: {100 * res_mean:.2f} +/- {100 * res_std:.2f}')

            if best_result is None or res_mean > best_result[0]:
                best_result = (res_mean, res_std)
                best_params = params
        if args.verbose:
            print(
                f'Best {model_type} Result: {100 * best_result[0]:.2f} +/- {100 * best_result[1]:.2f} with params {best_params}')
        return best_result, best_params

    def train_cross(self, data, grid_search=True, reduced=True):
        args = self.args
        if grid_search:
            gs_params = {
                'MLP': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                        'dropout': [0.0, 0.5]},
                'GCN': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                        'dropout': [0.0, 0.5]},
                'SGC': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                        'dropout': [0.0, 0.5], 'ntrans': [1, 2]},
                'APPNP': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                          'dropout': [0.05, 0.5], 'ntrans': [1, 2], 'alpha': [0.1, 0.2]},
                'Cheby': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                          'dropout': [0.0, 0.5]},
                'GraphSage': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                              'dropout': [0.0, 0.5]},
                'GAT': {'hidden': [16, 64], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                        'dropout': [0.05, 0.5, 0.7]}
            }
            # avoid OOM
            if args.dataset in ['reddit']:
                gs_params['GAT']['hidden'] = [8, 16]
            for model_type in gs_params:
                if reduced:
                    data.feat_syn, data.adj_syn, data.labels_syn = self.get_syn_data(model_type=model_type,
                                                                                     verbose=args.verbose)
                print(f'Starting Grid Search for {model_type}')
                best_result, best_params = self.grid_search(data, model_type, gs_params[model_type], reduced=reduced)
                args.logger.info(
                    f'Best {model_type} Result: {100 * best_result[0]:.2f} +/- {100 * best_result[1]:.2f} with params {best_params}')
        else:
            eval_model_list = ['MLP', 'GCN', 'SGC', 'APPNP', 'Cheby', 'GraphSage', 'GAT']
            evaluator = Evaluator(args)
            for model_type in eval_model_list:
                data.feat_syn, data.adj_syn, data.labels_syn = self.get_syn_data(model_type=model_type,
                                                                                 verbose=args.verbose)
                best_result = evaluator.evaluate(data, model_type=args.eval_model)
                args.logger.info(
                    f'{model_type} Result: {100 * best_result[0]:.2f} +/- {100 * best_result[1]:.2f}')

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
        model = eval(model_type)(data.feat_full.shape[1], args.hidden, data.nclass, args, mode=mode).to(
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

    def evaluate(self, data, model_type, verbose=True, reduced=True, mode='eval'):
        args = self.args

        data.feat_syn, data.adj_syn, data.labels_syn = self.get_syn_data(model_type=model_type, verbose=verbose)

        if verbose:
            print(f'evaluate reduced data by {model_type}')
            run_evaluation = trange(args.run_eval)
        else:
            run_evaluation = range(args.run_eval)

        res = []
        for i in run_evaluation:
            seed_everything(i)
            best_val_acc = self.test(data, model_type=model_type, verbose=False, reduced=reduced, mode=mode)
            res.append(best_val_acc)
            if verbose:
                run_evaluation.set_postfix(best_val_acc=best_val_acc)
        res = np.array(res)

        args.logger.info(f'Seed:{args.seed}, Test Mean Accuracy: {100 * res.mean():.2f} +/- {100 * res.std():.2f}')
        return res.mean(), res.std()

    def nas_evaluate(self, data, model_type, verbose=False, reduced=None):
        args = self.args
        res = []
        data.feat_syn, data.adj_syn, data.labels_syn = self.get_syn_data(model_type=model_type, verbose=verbose)
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
