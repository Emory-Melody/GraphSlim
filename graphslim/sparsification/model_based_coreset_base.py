import numpy as np

from graphslim.dataset.utils import save_reduced
from graphslim.evaluation import *
from graphslim.models import *
from graphslim.sparsification.coreset_base import CoreSet
from graphslim.utils import to_tensor


class MBCoreSet(CoreSet):
    def __init__(self, setting, data, args, **kwarg):
        super(MBCoreSet, self).__init__(setting, data, args, **kwarg)

    @verbose_time_memory
    def reduce(self, data, verbose=False, save=True):

        args = self.args
        if self.setting == 'trans':
            model = eval(args.eval_model)(data.feat_full.shape[1], args.eval_hidden, data.nclass, args, mode='eval').to(
                self.device)

            model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True, verbose=verbose,
                               setting=args.setting, reduced=False)

            model.test(data, setting=args.setting, verbose=True)
            embeds = model.predict(data.feat_full, data.adj_full).detach()

            idx_selected = self.select(embeds)

            data.adj_syn = data.adj_full[np.ix_(idx_selected, idx_selected)]
            data.feat_syn = data.feat_full[idx_selected]
            data.labels_syn = data.labels_full[idx_selected]

        if self.setting == 'ind':
            model = eval(args.eval_model)(data.feat_full.shape[1], args.eval_hidden, data.nclass, args, mode='eval').to(
                self.device)

            model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True, verbose=verbose,
                               setting=args.setting, reduced=False, reindex=True)

            model.test(data, setting=args.setting, verbose=True)

            model.eval()

            embeds = model.predict(data.feat_train, data.adj_train).detach()

            idx_selected = self.select(embeds)
            data.feat_syn = data.feat_train[idx_selected]
            data.adj_syn = data.adj_train[np.ix_(idx_selected, idx_selected)]
            data.labels_syn = data.labels_train[idx_selected]

        if verbose:
            print('selected nodes:', idx_selected.shape[0])
            print('induced edges:', data.adj_syn.sum())
        data.adj_syn, data.feat_syn, data.labels_syn = to_tensor(data.adj_syn, data.feat_syn, data.labels_syn,
                                                                 device='cpu')
        if save:
            save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)

        return data
