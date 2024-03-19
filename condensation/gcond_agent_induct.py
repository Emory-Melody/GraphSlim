import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from .utils import match_loss, regularization
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from models.gcn import GCN
from models.sgc import SGC
from models.sgc_multi import SGC as SGC1
from models.parametrized_adj import PGE
from torch_sparse import SparseTensor
from .gcond_agent_transduct import GCondBase


class GCondInd(GCondBase):
    def test_with_val(self, verbose=True):
        res = []

        data, device = self.data, self.device
        feat_syn, pge = self.feat_syn.detach(), self.pge
        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        dropout = 0.5 if self.args.dataset in ['reddit'] else 0
        model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=dropout,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device).to(device)

        adj_syn = pge.inference(feat_syn)
        args = self.args

        if args.save:
            torch.save(adj_syn, f'saved_ours/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
            torch.save(feat_syn, f'saved_ours/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')

        model.fit_with_val(feat_syn, adj_syn, data,
                           train_iters=600, normalize=True, verbose=False, noval=True, condensed=True)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        output = model.predict(data.feat_test, data.adj_test)

        loss_test = F.nll_loss(output, labels_test)
        acc_test = utils.accuracy(output, labels_test)
        res.append(acc_test.item())
        if verbose:
            print('Test Accuracy and Std:',
                  repr([res.mean(0), res.std(0)]))
        # print(adj_syn.sum(), adj_syn.sum() / (adj_syn.shape[0] ** 2))

        # if False:
        #     if self.args.dataset == 'ogbn-arxiv':
        #         thresh = 0.6
        #     elif self.args.dataset == 'reddit':
        #         thresh = 0.91
        #     else:
        #         thresh = 0.7
        #
        #     labels_train = torch.LongTensor(data.labels_train).cuda()
        #     output = model.predict(data.feat_train, data.adj_train)
        #     # loss_train = F.nll_loss(output, labels_train)
        #     # acc_train = utils.accuracy(output, labels_train)
        #     loss_train = torch.tensor(0)
        #     acc_train = torch.tensor(0)
        #     if verbose:
        #         print("Train set results:",
        #               "loss= {:.4f}".format(loss_train.item()),
        #               "accuracy= {:.4f}".format(acc_train.item()))
        #     res.append(acc_train.item())
        return res


5
