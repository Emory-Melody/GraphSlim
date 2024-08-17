from tqdm import trange

from graphslim.condensation.gcond_base import GCondBase
from graphslim.dataset.utils import *
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.utils import *
from graphslim.models import *
import pickle
import argparse
from io import StringIO
from pprint import pprint
from datetime import datetime
from importlib import import_module
# import pygcanl
import time

class Mirage(GCondBase):
    def __init__(self, setting, data, args, **kwargs):
        super(Mirage, self).__init__(setting, data, args, **kwargs)

    def append_to_file(self, message, filename):
        r"""This functions is used to log strings and general python objects
        in a pretty format to the log file specified by param filename."""
        if not isinstance(message, str):
            with StringIO() as stream:
                pprint(message, stream=stream)
                message = stream.getvalue()
        with open(filename, "a", encoding="utf-8") as output_file:
            output_file.write("-" * 20 + "\n")
            output_file.write(datetime.now().strftime("%d-%m-%Y %H:%M:%S\n"))
            output_file.write(message)
            output_file.write("\n")

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args
        pge = self.pge
        self.feat_syn, labels_syn = to_tensor(self.feat_syn, label=data.labels_syn, device=self.device)
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train,
                                              device=self.device)
        full_dataset = [Data(x=features, edge_index=adj, y=labels)]
        train_dataset = [Data(x=features, edge_index=adj, y=labels)]
        test_dataset = [Data(x=features, edge_index=adj, y=labels)]


        r"""The main function. Consider this the algorithm."""


        dataset_name = args.dataset
        args.run_num = 0
        output_filename = os.path.join(
            args.save_path, f"{dataset_name}_{args.run_num}_log.txt"
        )
        pprint(f"Outputting logs to {output_filename}")
        self.append_to_file({"seed": args.seed}, output_filename)
        self.append_to_file(args, output_filename)
        (
            proc_data,
            node_label_map_orig,
            node_label_map_full,
        ) = preprocess_dataset(train_dataset, full_dataset)

        args.n_hops = 2
        hops = args.n_hops
        self.append_to_file(f"n_hops (used for MPTree construction): {hops}", output_filename)
        time_0 = time.perf_counter()
        ret = pygcanl.canonical(proc_data, hops)
        time_1 = time.perf_counter()
        print(f"It took {time_1 - time_0:.4f}s to compute MPTree representations")
        self.append_to_file(
            f"It took {time_1 - time_0:.4f}s to compute MPTree representations",
            output_filename,
        )
        classes = [d.y.item() for d in proc_data]
        unique_classes = set(classes)
        prettified = [[prettify_canonical_label(tree) for tree in graph] for graph in ret]
        mapping = canonical_label_to_naturals(prettified)
        mapped_dataset = [[mapping[tree] for tree in graph] for graph in prettified]
        classwise_mapped_graphs = {
            class__: [
                graph for graph, class_ in zip(mapped_dataset, classes) if class_ == class__
            ]
            for class__ in unique_classes
        }
        classwise_trees = {0: set(), 1: set()}
        for class_ in (0, 1):
            for graph in classwise_mapped_graphs[class_]:
                for tree in graph:
                    classwise_trees[class_].add(tree)
        tree_class_count = tree_class_ctr(classes, mapped_dataset)
        invalid_tree_idx = get_invalid_trees(tree_class_count)
        invalid_tree_idx = {}
        selected_dataset = [
            list({tree for tree in graph if tree not in invalid_tree_idx})
            for graph in mapped_dataset
        ]
        classwise = {
            class__: [
                graph
                for graph, class_ in zip(selected_dataset, classes)
                if class_ == class__ and len(graph) > 0
            ]
            for class__ in unique_classes
        }
        # threshs = {0: 120, 1: 18}
        # threshs = {0: 144, 1: 18}
        # threshs = {0: 150, 1: 18}
        # threshs = {0: 60, 1: 10}
        threshs = args.threshs
        self.append_to_file(threshs, output_filename)
        patterns = pyfpgrowth_wrapper(classwise, threshs)
        # freq, pats = list(zip(*((v,k) for k,v in patterns[0].items())))
        # probs = np.asarray(freq)/np.sum(freq)
        # rng = np.random.RandomState(seed=vars(args).get('seed',0))
        # sampled_0 = rng.choice(np.arange(len(pats)), p=probs, size=len(patterns[1]), replace=False)
        # patterns[0] = {pats[k]:freq[k] for k in sampled_0}
        print(
            f"Unique trees in class 0: {len(classwise_trees[0])},"
            f" in class 1: {len(classwise_trees[1])}"
        )
        print(
            f"Patterns mined in class 0: {len(patterns[0])} (thresh: {threshs[0]}),"
            f" in class 1: {len(patterns[1])} (thresh: {threshs[1]})"
        )
        print(f"#(class 0)/#(class 1) = {len(patterns[0]) / len(patterns[1]):.2f}")
        self.append_to_file(
            f"Unique trees in class 0: {len(classwise_trees[0])},"
            f" in class 1: {len(classwise_trees[1])}",
            output_filename,
        )
        self.append_to_file(
            f"Patterns mined in class 0: {len(patterns[0])} (thresh: {threshs[0]}),"
            f" in class 1: {len(patterns[1])} (thresh: {threshs[1]})",
            output_filename,
        )
        self.append_to_file(
            f"#(class 0)/#(class 1) = {len(patterns[0]) / len(patterns[1]):.2f}",
            output_filename,
        )
        inv_mapping = {v: k for k, v in mapping.items()}
        reconstructed = {}
        # frequency is stored in patterns.
        # I need frequency with reconstructed sample mapping and
        # I need a dataset size to be generated.
        for class_ in unique_classes:
            class_reconstructed = []
            for idx, (pattern, freq) in enumerate(patterns[class_].items()):
                datas = []
                frq = {}
                for tree in pattern:
                    data = inv_mapping[tree]
                    (
                        node_label_map,
                        edge_label_map,
                        node_label_map_original,
                    ) = parse_canonical_label(data)
                    data = get_data(
                        node_label_map,
                        edge_label_map,
                        node_label_map_original,
                        node_label_map_orig,
                        edge_label_map_orig,
                        node_label_map_full,
                        class_,
                    )
                    datas.append(data)
                frq[idx] = {'freq': freq, 'data': datas}
                class_reconstructed.append(frq)
            reconstructed[class_] = class_reconstructed
        dataset = {}
        for class_, recon in reconstructed.items():
            class_dataset = []
            for rec in recon:
                idx = list(rec.keys())[0]
                data = disjointed_union(rec[idx]['data'], class_)
                if data is None:
                    # skipping when graph has only one node
                    continue
                data = roots_to_embed(data)
                rec[idx]['data'] = data
                class_dataset.append(rec)
            dataset[class_] = class_dataset
        #
        test_dataset_proc = preprocess_dataset_test(test_dataset, node_label_map_orig, edge_label_map_orig,
                                                    node_label_map_full)
        test_dataset_canonicalized = pygcanl.canonical(test_dataset_proc, hops)
        test_dataset_prettified = [[prettify_canonical_label(tree) for tree in graph] for graph in
                                   test_dataset_canonicalized]
        prettified_collapsed = set()
        [[[prettified_collapsed.add(inv_mapping[tree]) for tree in graph] for graph in dset.keys()] for dset in
         patterns.values()]
        test_dataset_prettified = [[tree for tree in graph if tree in prettified_collapsed] for graph in
                                   test_dataset_prettified]
        test_dataset_reconstructed = []
        assert len(test_dataset_prettified) == len(test_dataset)
        for trees, data in zip(test_dataset_prettified, test_dataset):
            cls = data.y.item()
            if len(trees) == 0:
                datum = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y,
                             roots_to_embed=torch.ones(data.x.shape[0]))
            else:
                datum = []
                for tree in trees:
                    node_label_map, edge_label_map, node_label_map_original = parse_canonical_label(tree)
                    tree_ = get_data(node_label_map, edge_label_map, node_label_map_original, node_label_map_orig,
                                     edge_label_map_orig, node_label_map_full, cls)
                    datum.append(tree_)
                datum = disjointed_union(datum, cls)
            test_dataset_reconstructed.append(datum)
        assert not any(hasattr(d, 'node_attr') for d in test_dataset_reconstructed)
        with open(f"{args.output_dir}/saved_dataset_{args.run_num}.pkl", "wb") as data_file:
            pickle.dump(dataset, data_file)






        # append_to_file("Turning off frequency based sampling for comparison", output_filename)


        if it in args.checkpoints:
            self.adj_syn = adj_syn_inner
            data.adj_syn, data.feat_syn, data.labels_syn = self.adj_syn.detach(), self.feat_syn.detach(), labels_syn.detach()
            best_val = self.intermediate_evaluation(best_val, loss_avg)

        return data
