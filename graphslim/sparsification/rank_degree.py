import networkit as nk
import numpy as np
import torch
from graphslim.sparsification.edge_sparsification_base import EdgeSparsifier
from torch_geometric.utils.convert import from_networkit

import random


class RankDegree(EdgeSparsifier):
    def __init__(self, setting, data, args, **kwargs):
        super(RankDegree, self).__init__(setting, data, args, **kwargs)

    def edge_cutter(self, G):
        args = self.args
        rho = 0.1
        targetNum = int(G.numberOfEdges() * args.reduction_rate)
        print(f"\ntargetNum: {targetNum}")
        seeds = set()
        seen = set()

        new_G = nk.graph.Graph(G.numberOfNodes(), weighted=G.isWeighted(), directed=G.isDirected())
        iter_count, count = 0, 0
        while new_G.numberOfEdges() < targetNum:
            iter_count += 1
            # print(f"new_G.numberOfEdges(): {new_G.numberOfEdges()}")
            if not seeds:
                # generate random seeds from all_node_set
                seeds = random.sample(list(range(0, G.numberOfNodes())), 10)
                seeds = set(seeds)

            if count > int(0.01 * targetNum):
                iter_count = 0
                count = 0
            elif iter_count > 1000 and count <= int(
                    0.01 * targetNum) and rho < 0.99:  # less than 10 edges added in the last 1000 iterations
                iter_count = 0
                rho += 0.1
                seen = set()
                count = 0
                print(f"increasing rho: {rho:.3}, new_G.numberOfEdges(): {new_G.numberOfEdges()}")

            new_seeds = set()
            for seed in seeds:
                if seed in seen:
                    continue
                seen.add(seed)
                # get neighbors
                neighbors = list(G.iterNeighbors(seed))
                # rank neighbors by degree
                neighbors = sorted(neighbors, key=lambda x: G.degree(x), reverse=True)
                # select top rho * len(neighbors)
                neighbors = neighbors[:max(1, int(len(neighbors) * rho))]
                # add edges
                for neighbor in neighbors:
                    if not new_G.hasEdge(seed, neighbor):
                        if G.isWeighted():
                            new_G.addEdge(seed, neighbor, w=G.weight(seed, neighbor))
                        else:
                            new_G.addEdge(seed, neighbor)
                        new_seeds.add(neighbor)
                        count += 1
            seeds = new_seeds
        if args.verbose:
            nk.overview(new_G)
        edge_index, edge_attr = from_networkit(new_G)

        return edge_index, edge_attr
