import numpy as np
import torch.nn.functional as F
from tqdm import trange

from graphslim.dataset import *
from graphslim.models import *
from graphslim.utils import accuracy, seed_everything, is_sparse_tensor


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
    # def test_gat(self, nlayers, model_type, verbose=False):
    #     res = []
    #     args = self.args
    #
    #     if args.dataset in ['cora', 'citeseer']:
    #         args.epsilon = 0.5  # Make the graph sparser as GAT does not work well on dense graph
    #     else:
    #         args.epsilon = 0.01
    #
    #     print('======= testing %s' % model_type)
    #     data, device = self.data, self.device
    #
    #     feat_syn, adj_syn, labels_syn = self.get_syn_data(model_type)
    #     # with_bn = True if self.args.dataset in ['ogbn-arxiv'] else False
    #     with_bn = False
    #     if model_type == 'GAT':
    #         model = GAT(nfeat=feat_syn.shape[1], nhid=16, heads=16, dropout=0.0,
    #                     weight_decay=0e-4, nlayers=self.args.nlayers, lr=0.001,
    #                     nclass=data.nclass, device=device, dataset=self.args.dataset).to(device)
    #
    #     noval = True if args.dataset in ['reddit', 'flickr'] else False
    #     model.fit(feat_syn, adj_syn, labels_syn, np.arange(len(feat_syn)), noval=noval, data=data,
    #               train_iters=10000 if noval else 3000, normalize=True, verbose=verbose)
    #
    #     model.eval()
    #     labels_test = torch.LongTensor(data.labels_test).to(args.device)
    #
    #     if args.dataset in ['reddit', 'flickr']:
    #         output = model.predict(data.feat_test, data.adj_test)
    #         loss_test = F.nll_loss(output, labels_test)
    #         acc_test = utils.accuracy(output, labels_test)
    #         res.append(acc_test.item())
    #         if verbose:
    #             print("Test set results:",
    #                   "loss= {:.4f}".format(loss_test.item()),
    #                   "accuracy= {:.4f}".format(acc_test.item()))
    #
    #     else:
    #         # Full graph
    #         output = model.predict(data.feat_full, data.adj_full)
    #         loss_test = F.nll_loss(output[data.idx_test], labels_test)
    #         acc_test = utils.accuracy(output[data.idx_test], labels_test)
    #         res.append(acc_test.item())
    #         if verbose:
    #             print("Test set results:",
    #                   "loss= {:.4f}".format(loss_test.item()),
    #                   "accuracy= {:.4f}".format(acc_test.item()))
    #
    #     labels_train = torch.LongTensor(data.labels_train).to(args.device)
    #     output = model.predict(data.feat_train, data.adj_train)
    #     loss_train = F.nll_loss(output, labels_train)
    #     acc_train = utils.accuracy(output, labels_train)
    #     if verbose:
    #         print("Train set results:",
    #               "loss= {:.4f}".format(loss_train.item()),
    #               "accuracy= {:.4f}".format(acc_train.item()))
    #     res.append(acc_train.item())
    #     return res

    def get_syn_data(self, model_type=None, verbose=False):

        args = self.args
        adj_syn, feat_syn, labels_syn = load_reduced(args, args.valid_result)
        if is_sparse_tensor(adj_syn):
            adj_syn = adj_syn.to_dense()

        if model_type == 'MLP':
            adj_syn = adj_syn - adj_syn

        if verbose:
            print('Sum:', adj_syn.sum().item(), (adj_syn.sum() / (adj_syn.shape[0] ** 2)).item())
        # setting threshold to sparsify synthetic graph
        if args.method in ['gcond', 'doscond']:
            print('Sparsity:', adj_syn.nonzero().shape[0] / (adj_syn.shape[0] ** 2))
            if args.threshold > 0:
                adj_syn[adj_syn < args.threshold] = 0
                if verbose:
                    print('Sparsity after truncating:', adj_syn.nonzero().shape[0] / (adj_syn.shape[0] ** 2))
            # else:
            #     print("structure free methods do not need to truncate the adjacency matrix")

        return feat_syn, adj_syn, labels_syn

    def test(self, data, model_type, verbose=True, reduced=True):
        args = self.args
        res = []
        feat_syn, adj_syn, labels_syn = data.feat_syn, data.adj_syn, data.labels_syn

        if verbose:
            print('======= testing %s' % model_type)
        # if model_type == 'MLP':
        #     data.adj_syn = data.adj_syn - data.adj_syn
        #     model_class = GCN
        # else:
        #     model_class = eval(model_type)

        model = eval(model_type)(feat_syn.shape[1], args.eval_hidden, data.nclass, args, mode='eval').to(self.device)

        model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True, verbose=verbose,
                           setting=args.setting,
                           reduced=reduced)

        model.eval()
        labels_test = data.labels_test.long().to(args.device)

        # if model_type == 'MLP':
        #     output = model.predict(data.feat_test, sp.eye(len(data.feat_test),normadj))

        if args.setting == 'ind':
            output = model.predict(data.feat_test, data.adj_test)
            loss_test = F.nll_loss(output, labels_test)
            acc_test = accuracy(output, labels_test)
            res.append(acc_test.item())
            if verbose:
                print("Test set results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

        # if not args.dataset in ['reddit', 'flickr']:
        else:
            # Full graph
            output = model.predict(data.feat_full, data.adj_full)
            loss_test = F.nll_loss(output[data.idx_test], labels_test)
            acc_test = accuracy(output[data.idx_test], labels_test)
            res.append(acc_test.item())
            if verbose:
                print("Test full set results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

            # labels_train = torch.LongTensor(data.labels_train).cuda()
            # output = model.predict(data.feat_train, data.adj_train)
            # loss_train = F.nll_loss(output, labels_train)
            # acc_train = accuracy(output, labels_train)
            # if verbose:
            #     print("Train set results:",
            #           "loss= {:.4f}".format(loss_train.item()),
            #           "accuracy= {:.4f}".format(acc_train.item()))
            # res.append(acc_train.item())
        return res

    def train_cross(self, data):
        args = self.args
        data.nclass = data.nclass.item()

        for model_type in ['MLP', 'GCN', 'SGC', 'APPNP', 'Cheby']:  # 'GraphSage'
            data.feat_syn, data.adj_syn, data.labels_syn = self.get_syn_data(model_type=model_type,
                                                                             verbose=args.verbose)
            res = []
            if args.verbose:
                run_evaluation = trange(args.run_evaluation)
            else:
                run_evaluation = range(args.run_evaluation)
            for i in run_evaluation:
                seed_everything(args.seed + i)
                res.append(self.test(data, model_type=model_type, verbose=False, reduced=True))
            res = np.array(res)
            res_mean, res_std = res.mean(), res.std()
            print(f'{model_type} Test Mean Result: {100 * res_mean:.2f} +/- {100 * res_std:.2f}')

        # print('=== testing GAT')
        # res = []
        # nlayer = 2
        # for i in range(runs):
        #     res.append(self.test_gat(verbose=True, nlayers=nlayer, model_type='GAT'))
        # res = np.array(res)
        # print('Layer:', nlayer)
        # print('Test/Full Test/Train Mean Accuracy:',
        #         repr([res.mean(0), res.std(0)]))
        # final_res['GAT'] = [res.mean(0), res.std(0)]

        # print('Final result:', final_res)

    def evaluate(self, data, model_type, verbose=True, reduced=True):
        # model_type: ['GCN1', 'GraphSage', 'SGC1', 'MLP', 'APPNP1', 'Cheby']
        # self.data = data
        args = self.args

        res = []
        data.feat_syn, data.adj_syn, data.labels_syn = self.get_syn_data(model_type, verbose=args.verbose)

        if verbose:
            run_evaluation = trange(args.run_evaluation)
        else:
            run_evaluation = range(args.run_evaluation)
        for i in run_evaluation:
            seed_everything(args.seed + i)
            res.append(self.test(data, model_type=model_type, verbose=verbose, reduced=reduced))
        res = np.array(res)

        if verbose:
            print(f'Test Mean Accuracy: {100 * res.mean():.2f} +/- {100 * res.std():.2f}')
        return res.mean(), res.std()

    def nas_evaluate(self, data, model_type, verbose=True, reduced=None):
        args = self.args
        res = []
        data.feat_syn, data.adj_syn, data.labels_syn = self.get_syn_data(model_type, verbose=args.verbose)
        if verbose:
            run_evaluation = trange(args.run_evaluation)
        else:
            run_evaluation = range(args.run_evaluation)
        for i in run_evaluation:
            seed_everything(args.seed + i)
            if verbose:
                print('======= testing %s' % model_type)
            model_class = eval(model_type)

            feat_syn, adj_syn, labels_syn = data.feat_syn, data.adj_syn, data.labels_syn

            if reduced:
                model = model_class(nfeat=data.x.shape[1], nhid=args.eval_hidden, nclass=data.nclass,
                                    nlayers=args.nlayers,
                                    dropout=0, lr=args.lr_test, weight_decay=5e-4, device=self.device,
                                    activation=args.activation, alpha=args.alpha).to(self.device)

                best_acc_val = model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True,
                                                  verbose=verbose,
                                                  setting=args.setting,
                                                  reduced=True)
            else:
                model = model_class(nfeat=feat_syn.shape[1], nhid=args.eval_hidden, nclass=data.nclass,
                                    nlayers=args.nlayers,
                                    dropout=0, lr=args.lr_test, weight_decay=5e-4, device=self.device,
                                    activation=args.activation, alpha=args.alpha).to(self.device)

                best_acc_val = model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True,
                                                  verbose=verbose,
                                                  setting=args.setting)
            res.append(best_acc_val.item())
        res = np.array(res)

        if verbose:
            print(f'Test Mean Accuracy: {100 * res.mean():.2f} +/- {100 * res.std():.2f}')
        return res.mean(), res.std()
