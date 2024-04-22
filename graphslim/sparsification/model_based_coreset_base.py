import time

import numpy as np

from graphslim.dataset.utils import save_reduced
from graphslim.models import *
from graphslim.sparsification.coreset_base import CoreSet
from graphslim.utils import to_tensor, getsize_mb


class MBCoreSet(CoreSet):

    def reduce(self, data, verbose=False):
        if verbose:
            start = time.perf_counter()
        args = self.args
        if self.setting == 'trans':
            model = GCN(nfeat=data.feat_full.shape[1], nhid=args.hidden, nclass=data.nclass, device=args.device,
                        weight_decay=args.weight_decay).to(args.device)
            model.fit_with_val(data, train_iters=args.eval_epochs, verbose=verbose, setting='trans')
            # model.test(data, verbose=True)
            embeds = model.predict(data.feat_full, data.adj_full).detach()

            idx_selected = self.select(embeds)

            data.adj_syn = data.adj_full[np.ix_(idx_selected, idx_selected)]
            data.feat_syn = data.feat_full[idx_selected]
            data.labels_syn = data.labels_full[idx_selected]

        if self.setting == 'ind':
            model.fit_with_val(data, train_iters=args.eval_epochs, verbose=verbose, setting='ind', reindex=True)

            model.eval()

            embeds = model.predict(data.feat_train, data.adj_train).detach()

            idx_selected = self.select(embeds)
            data.feat_syn = data.feat_train[idx_selected]
            data.adj_syn = data.adj_train[np.ix_(idx_selected, idx_selected)]
            data.labels_syn = data.labels_train[idx_selected]

        print('selected nodes:', idx_selected.shape[0])
        print('induced edges:', data.adj_syn.sum())
        data.adj_syn, data.feat_syn, data.labels_syn = to_tensor(data.adj_syn, data.feat_syn, data.labels_syn,
                                                                 device='cpu')
        save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)
        if verbose:
            end = time.perf_counter()
            runTime = end - start
            runTime_ms = runTime * 1000
            print("Reduce Time: ", runTime, "s")
            print("Reduce Time: ", runTime_ms, "ms")
            if args.setting == 'trans':
                origin_storage = getsize_mb([data.x, data.edge_index, data.y])
            else:
                origin_storage = getsize_mb([data.feat_train, data.adj_train, data.labels_train])
            condensed_storage = getsize_mb([data.feat_syn, data.adj_syn, data.labels_syn])
            print(f'Origin graph:{origin_storage:.2f}Mb  Condensed graph:{condensed_storage:.2f}Mb')

        return data
