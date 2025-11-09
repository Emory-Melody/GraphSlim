import os
import torch
import numpy as np
from numpy import random
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import to_undirected, subgraph
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from collections import Counter
from tqdm import trange
from timeit import default_timer as timer
from torch_scatter import scatter_add
from graphslim.utils import to_tensor, normalize_adj_tensor
from graphslim.dataset.utils import save_reduced, load_reduced
import torch.nn.functional as F
from graphslim.condensation.gcond_base import GCondBase
import skfuzzy as fuzz
import concurrent.futures
from sklearn.metrics import pairwise_distances  # Faster than cdist





class GECC(GCondBase):
    """
    ECA (Evolving Clustering Aggregation) is a variant of Coarsen that performs
    clustering on aggregated features for both transductive and inductive settings.
    It outputs synthesized features (`feat_syn`) and labels (`label_syn`).
    """

    def reduce(self, data, verbose=True, save=True):
        """
        Reduces the data by aggregating features using a pre-convolution step and clustering.

        Parameters
        ----------
        data : TransAndInd
            The data to be reduced.
        verbose : bool, optional
            If True, prints verbose output. Defaults to True.
        save : bool, optional
            If True, saves the reduced data. Defaults to True.

        Returns
        -------
        TransAndInd
            The reduced data with synthesized features and labels.
        """
        args = self.args
        device = torch.device(args.device)
        start_total = timer()
        alpha, beta, gamma = args.agg_alpha, args.agg_beta, args.agg_gamma

        ####################################################
        # 1) Pre-aggregate features depending on the setting
        ####################################################

        if args.setting == "trans":
            if args.dataset in ["ogbn-products"]:
                # Move full feature matrix to GPU
                data.feat_full = data.feat_full.to(device)

                # Initialize aggregated feature tensor
                num_nodes, feature_dim = data.feat_full.size()
                final_feat_agg = torch.zeros((num_nodes, feature_dim), device=device)

                # Initialize NeighborSampler with configurable depth
                train_loader = NeighborSampler(
                    edge_index=data.edge_index,
                    sizes=[15] * args.depth,  # Number of neighbors for each hop
                    batch_size=1024,
                    shuffle=True,
                    num_workers=4,
                )

                # Pre-compute weights for each hop
                weights = [gamma, alpha, beta]  # For first 3 hops
                for _ in range(3, args.depth+1):
                    weights.append(0.5)  # Use 1.0 for deeper layers

                # Aggregate features using NeighborSampler
                for batch_size, n_id, adjs in train_loader:
                    # Move all adjacency matrices to GPU at once
                    adjs = [adj.to(device) for adj in adjs]
                    
                    # Initialize feature tensors for each hop
                    feat_hops = [data.feat_full[n_id]]  # 0-hop features
                    
                    # Compute features for each hop
                    for i, adj in enumerate(adjs):
                        row, col = adj.edge_index
                        # Use scatter_add for efficient aggregation
                        feat_next = scatter_add(
                            feat_hops[-1][row], 
                            col, 
                            dim=0, 
                            dim_size=feat_hops[0].size(0)
                        )
                        feat_hops.append(feat_next)

                    # Combine features with weights efficiently
                    feat_agg = torch.zeros_like(feat_hops[0])
                    for i, (weight, feat) in enumerate(zip(weights, feat_hops)):
                        feat_agg.add_(weight * feat)
                    
                    # Update final features in-place
                    final_feat_agg[n_id] = feat_agg

            else:
                # Other transductive datasets
                data.adj_fully = to_tensor(data.adj_full, device=args.device)
                data.first_hop = normalize_adj_tensor(data.adj_fully, sparse=True)

                # Pre-compute weights for each hop
                weights = [gamma, alpha, beta]  # For first 3 hops
                for _ in range(3, args.depth+1):
                    weights.append(0.5)  # Use 1.0 for deeper layers

                # Initialize feature tensors
                feat_hops = [to_tensor(data.feat_full, device=device)]
                
                # Compute features for each hop
                for i in range(1, args.depth+1):
                    feat_next = matmul(data.first_hop, feat_hops[-1]).float()
                    feat_hops.append(feat_next)

                # Combine features with weights efficiently
                final_feat_agg = torch.zeros_like(feat_hops[0])
                for weight, feat in zip(weights, feat_hops):
                    final_feat_agg.add_(weight * feat)

            if verbose:
                end_agg = timer()
                args.logger.info(
                    f"=== Finished trans PreAggregation in {end_agg - start_total:.2f} sec ==="
                )

        else:
            # Inductive setting requires local aggregation on the subgraph
            data.feat_train = data.feat_train.to(device)
            data.adj_train = to_tensor(data.adj_train, device=device)

            data.first_hop = normalize_adj_tensor(data.adj_train, sparse=True)

            # Pre-compute weights for each hop
            weights = [gamma, alpha, beta]  # For first 3 hops
            for _ in range(3, args.depth):
                weights.append(1.0)  # Use 2.0 for deeper layers

            # Initialize feature tensors
            feat_hops = [to_tensor(data.feat_train, device=device)]
            
            # Compute features for each hop
            for i in range(1, args.depth):
                feat_next = matmul(data.first_hop, feat_hops[-1]).float()
                feat_hops.append(feat_next)

            # Combine features with weights efficiently
            final_feat_agg = torch.zeros_like(feat_hops[0])
            for weight, feat in zip(weights, feat_hops):
                final_feat_agg.add_(weight * feat)

        if verbose:
            end_agg = timer()
            args.logger.info(
                f"=== Finished ind PreAggregation in {end_agg - start_total:.2f} sec ==="
            )

        ###########################
        # 2) Prepare training data
        ###########################
        if hasattr(data, "labels_syn") and data.labels_syn is not None:
            # Existing synthetic labels
            y_syn = data.labels_syn
            self.labels_train = data.labels_train
            y_train = data.labels_train
            if isinstance(y_syn, torch.Tensor):
                data.n_classes = len(np.unique(y_syn.detach().cpu().numpy()))
            else:
                data.n_classes = len(np.unique(y_syn))
            if verbose:
                print(f"#classes in training set: {data.n_classes}")
            # Create syn_class_indices for existing labels
            syn_class_indices = {}
            pos = 0
            for c in range(data.n_classes):
                count = sum(y_syn == c)
                syn_class_indices[c] = (pos, pos + count)
                pos += count
        else:
            # Prepare synthetic labels
            if verbose:
                print("Start preparing select")
            y_syn, y_train, syn_class_indices = self.prepare_select(data, args)

        # Define training features based on setting
        if args.setting == "trans":
            x_train = final_feat_agg[data.train_mask]  # (N_train, d)
        else:
            x_train = final_feat_agg  # (N_train_sub, d)

        # Basic sanity checks
        y_train = data.labels_train
        data.nclass = (
            len(torch.unique(y_train)) if not hasattr(data, "nclass") else data.nclass
        )
        if verbose:
            print(x_train.shape)
            print(y_train.shape)

        ######################################################
        # 3) Perform Class-Balanced Clustering
        ######################################################
        # Initialize centroids storage

        # if args.evolve and args.si != '1':
        #     # Attempt to derive the "previous" split index (e.g., from "1+2+3" to "1+2")
        #     current_si = args.si
        #     prev_si = get_previous_split(args.si)  # Derive the previous split index

        #     if prev_si:  # If a previous split exists
        #         # Update args.save_path to replace current_si with prev_si
        #         args.save_path = args.save_path.replace(args.si, prev_si)
                
        #         # Temporarily update args.si and load data for the previous split
        #         args.si = prev_si
        #         _, pre_feat, pre_labels = load_reduced(args)

        #         # Build centroids_per_class using data from the previous split
        #         print("Using previous centroids")
        #         centroids_per_class = {
        #             c: pre_feat[pre_labels == c] for c in range(data.nclass)
        #         }

        #         # Restore the current split index and save path
        #         args.si = current_si
        #         args.save_path = args.save_path.replace(prev_si, current_si)
        #     else:
        #         # If there's no previous split (e.g., si = "1"), initialize centroids as None
        #         print("No previous split found. Initializing centroids as None.")
                # centroids_per_class = {c: None for c in range(data.nclass)}
        # else:
        # If evolve is not enabled or si is "1", initialize centroids as None
        print("Evolve is not enabled or si is '1'. Initializing centroids as None.")
        centroids_per_class = {c: None for c in range(data.nclass)} 
        synthetic_labels = []
        synthetic_features = []

        for c in range(data.nclass):
            # Extract features for class c
            class_mask = y_train == c
            x_c = x_train[class_mask]

            if x_c.size(0) == 0:
                if verbose:
                    print(f"No samples for class {c}. Skipping.")
                continue

            # Determine number of clusters for class c
            # This can be a fixed number or based on some criteria
            # Here, we use a parameter `num_clusters_per_class`
            n_c = sum(y_syn == c)

            # Move to CPU and convert to numpy for clustering
            x_c_np = x_c.cpu().numpy()

            # Perform clustering
            # updated_centroids_c = perform_balance_evolve_clusteringjk(
            #     x_c_current=x_c_np,
            #     y_syn=None,  # Not used here
            #     c=c,
            #     n_c=n_c,
            #     x_syn_c=centroids_per_class[c],
            #     args=args,
            #     data=data,
            # )
            updated_centroids_c = perform_balance_evolve_clusteringjk(
                x_c_current=x_c_np,
                y_syn=y_syn,
                c=c,
                x_syn_c=centroids_per_class[c],
                args=args,
                n_c=n_c,
                epsilon=1e-8,
                max_iters=300,
                save_p=False,
                class_id=c
            )
              
            if updated_centroids_c is None:
                raise ValueError(f"No centroids returned for class {c}.")

            if updated_centroids_c.shape[0] != n_c:
                raise ValueError(
                    f"Number of returned centroids {updated_centroids_c.shape[0]} != n_c={n_c} for class {c}."
                )

            # Update centroids storage
            centroids_per_class[c] = updated_centroids_c

            # Collect centroids and labels
            synthetic_features.append(updated_centroids_c.clone().detach())
            synthetic_labels.append(
                torch.full((n_c,), c, dtype=torch.long, device=device)
            )

        # Combine all synthetic features and labels
        if not synthetic_features:
            # Handle the case where no synthetic features were generated
            # Save empty tensors to indicate that no reduction was performed
            if save:
                # Create empty tensors with the correct device
                x_syn = torch.empty(0, data.feat_full.shape[1], device=device)
                y_syn_incremental = torch.empty(0, dtype=torch.long, device=device)
                adj_syn = torch.empty(0, 0, device=device)
                
                # Save the empty tensors
                save_reduced(adj_syn, x_syn, y_syn_incremental, args)
                if verbose:
                    args.logger.warning("No synthetic features were generated. Saving empty graph.")
            return data

        if synthetic_features:
            x_syn = torch.cat(synthetic_features, dim=0)
            y_syn_incremental = torch.cat(synthetic_labels, dim=0)

            # Create adjacency for synthetic set (identity matrix)
            adj_syn = torch.eye(x_syn.shape[0], device=device)

            # Create assignment matrix (maps synthetic nodes to original training nodes)
            print(f"Creating assignment matrix with shape ({x_syn.shape[0]}, {data.feat_full.shape[0]})")
            assignment = torch.zeros((x_syn.shape[0], data.feat_full.shape[0]), device=device)
            
            # Print class statistics to understand assignments better
            for i in range(data.nclass):
                class_indices = torch.where(data.labels_train == i)[0]
                print(f"Class {i}: {len(class_indices)} original training nodes, {sum(y_syn_incremental == i).item()} synthetic nodes")
            
            # Check if using Fuzzy C-Means (fuzziness > 1.0)
            if args.fuzziness > 1.0:
                print(f"Using top-1 distance for assignment (Fuzzy C-Means with m={args.fuzziness})")
                
                # For each class, compute distances and create assignment based on closest nodes
                for i in range(data.nclass):
                    # Get indices for this class
                    class_indices = torch.where(data.labels_train == i)[0]
                    syn_indices = torch.where(y_syn_incremental == i)[0]
                    
                    if len(syn_indices) == 0 or len(class_indices) == 0:
                        continue
                    
                    # Get features for synthetic and original nodes
                    syn_features = x_syn[syn_indices]
                    orig_features = data.feat_train[data.labels_train == i]
                    
                    # Make sure both tensors are on the same device
                    if syn_features.device != orig_features.device:
                        print(f"Moving orig_features from {orig_features.device} to {syn_features.device}")
                        orig_features = orig_features.to(syn_features.device)
                    
                    # Compute pairwise distances between synthetic and original nodes
                    # [syn_nodes, orig_nodes]
                    dist_matrix = torch.cdist(syn_features, orig_features, p=2)
                    
                    # For each synthetic node, find the closest original node
                    closest_indices = torch.argmin(dist_matrix, dim=1)
                    
                    # Assign each synthetic node to its closest original node
                    for j, syn_idx in enumerate(syn_indices):
                        orig_idx = class_indices[closest_indices[j]]
                        assignment[syn_idx, orig_idx] = 1.0
                        
                    # Print the first few assignments for debugging
                    for j in range(min(5, len(syn_indices))):
                        syn_idx = syn_indices[j]
                        orig_idx = class_indices[closest_indices[j]]
                        print(f"Synthetic node {syn_idx} (class {i}) assigned to original node {orig_idx}")
                        
                print("Assignment matrix created using top-1 distances")
            else:
                # Original method: uniform assignment across all nodes of the same class
                print("Using uniform assignment across all nodes of the same class")
                # For each class, assign synthetic nodes to all original nodes of that class
                for i, (start, end) in enumerate(syn_class_indices.values()):
                    # Get the original indices for this class
                    class_indices = torch.where(data.labels_train == i)[0]
                    print(f"Creating assignments for class {i}: {end-start} synthetic nodes mapped to {len(class_indices)} original nodes")
                    
                    # Assign each synthetic node to its corresponding original nodes
                    for j in range(start, end):
                        assignment[j, class_indices] = 1.0 / len(class_indices)
                    
                    # Print the first assignment for debugging
                    if start < end:
                        print(f"Synthetic node {start} (class {i}) uniformly assigned to {len(class_indices)} original nodes")
                        assigned_original = torch.where(assignment[start] > 0)[0]
                        if isinstance(assigned_original, torch.Tensor):
                            first_few = assigned_original[:5].cpu().tolist()
                        else:
                            first_few = assigned_original[:5].tolist() if hasattr(assigned_original, 'tolist') else list(assigned_original[:5])
                        print(f"  Sample of assigned original nodes: {first_few}...")
            
            # Print information about the assignment matrix
            print(f"Assignment matrix created with shape {assignment.shape}")
            non_zero_counts = (assignment > 0).sum(dim=1)
            print(f"Each synthetic node is connected to {non_zero_counts.float().mean().item():.2f} original nodes on average")
            print(f"Min connections: {non_zero_counts.min().item()}, Max connections: {non_zero_counts.max().item()}")
            
            # Verify assignments for the first few synthetic nodes
            for i in range(min(5, assignment.shape[0])):
                assigned_original = torch.where(assignment[i] > 0)[0]
                print(f"Synthetic node {i} (class {y_syn_incremental[i].item()}) has {len(assigned_original)} connections")
                if len(assigned_original) > 0:
                    if isinstance(assigned_original, torch.Tensor):
                        first_few = assigned_original[:5].cpu().tolist()
                    else:
                        first_few = assigned_original[:5].tolist() if hasattr(assigned_original, 'tolist') else list(assigned_original[:5])
                    print(f"  Connected to original nodes: {first_few}...")

            # Assign to data object
            data.feat_syn = x_syn
            data.labels_syn = y_syn_incremental
            data.adj_syn = adj_syn
            data.assignment = assignment  # Store the assignment matrix

            # Save reduced data if required
            if save:
                save_reduced(adj_syn, x_syn, y_syn_incremental, args)
                if verbose:
                    print(x_syn.shape, y_syn_incremental.shape, adj_syn.shape)
                # self.intermediate_evaluation(best_val=0, save=False)
                if verbose:
                    print("Saved synthetic data.")

        if verbose:
            end_total = timer()
            args.logger.info(
                f"=== Finished ECA Reduction in {end_total - start_total:.2f} sec ==="
            )

        return data

    def prepare_select(self, data, args):
        """
        Prepares and selects synthetic labels and features for coarsening.

        Parameters
        ----------
        data : TransAndInd
            The data to be processed.
        args : object
            Arguments containing various settings for the coarsening process.

        Returns
        -------
        tuple
            A tuple containing:
            - labels_syn : ndarray
                Synthesized labels.
            - labels_train : tensor
                Training labels.
            - syn_class_indices : dict
                Dictionary mapping class indices to their start and end positions in labels_syn.
        """

        labels_train = data.labels_train.numpy()
        counter = Counter(labels_train.tolist())

        # Sort classes by their frequency (ascending)
        sorted_counter = sorted(counter.items(), key=lambda x: x[1])

        # Keep track of where each class will reside in labels_syn if needed
        syn_class_indices = {}

        # We'll create a list of arrays and then concatenate once,
        # which is more efficient than extending lists in every iteration.
        labels_syn_list = []

        pos = 0
        for c, freq in sorted_counter:
            # Compute how many instances of this label to keep
            count = max(int(freq * args.reduction_rate), 1)

            # Record the start and end indices for the current class
            syn_class_indices[c] = (pos, pos + count)

            # Create a NumPy array of repeated label c, append to list
            labels_syn_list.append(np.full(count, c, dtype=labels_train.dtype))

            # Update position
            pos += count

        labels_syn = np.concatenate(labels_syn_list)

        return labels_syn, data.labels_train, syn_class_indices


# def perform_balance_evolve_clusteringjk(
#     x_c_current: np.ndarray,  # New data for class c, shape (#points, d)
#     y_syn,  # Not used here, reserved for future expansions
#     c,  # Class label (unused here, can be used for logging)
#     x_syn_c,  # Previous centroids (Tensor) or None
#     args,
#     n_c: int,  # Number of clusters for class c
#     n_old=None,  # Membership counts (or other measure) for old centroids
#     alpha: float = 1,  # Sub-linear exponent for punishing large clusters
#     epsilon: float = 1e-5,  # Convergence threshold
#     max_iters: int = 100,  # Max iterations (stop if not converged)
# ) -> torch.Tensor:
#     """
#     Incremental (batch/streaming) clustering supporting Weighted K-Means and Fuzzy C-Means.
#
#     1) "Punish" large clusters:
#        - We use a sub-linear weighting function w = n_old^alpha for old centroids,
#          preventing very large clusters from dominating.
#
#     2) "Inherit" old centroids for efficiency:
#        - Old centroids are treated like single points with weight = w.
#        - Combine them with new data (weight=1 each) in one Weighted K-Means step.
#
#     3) Fuzzy or Hard Clustering:
#        - If args.fuzziness == 1.0 => Weighted K-Means
#        - If args.fuzziness > 1.0  => Fuzzy C-Means (not shown in detail here)
#
#     Args:
#       x_c_current (np.ndarray):  New data for class c, shape [#points, d].
#       y_syn:                     (Not used in this snippet, placeholder)
#       c:                         Class label (unused here, but can be used for logging).
#       x_syn_c (torch.Tensor):    Old centroids, shape [old_count, d], or None if none exist.
#       args:                      Contains device, seed, fuzziness, etc.
#       data:                      (Not used in this snippet)
#       n_c (int):                 Desired #clusters for class c.
#       n_old (torch.Tensor):      For each old centroid, how many points it represented (size = old_count).
#       alpha (float):             Exponent for sub-linear weighting, e.g. 0.5 => sqrt.
#       epsilon (float):           Convergence threshold for updates.
#       max_iters (int):           Max #iterations to avoid infinite loop.
#
#     Returns:
#       torch.Tensor of shape [n_c, d] with updated centroids.
#     """
#     device = torch.device(args.device)
#
#     # Convert new data to torch
#     x_c_torch = torch.tensor(x_c_current, dtype=torch.float32, device=device)
#
#     with torch.random.fork_rng():
#         # For reproducibility
#         torch.manual_seed(args.seed)
#
#         # -----------------------------------------------------------
#         # Step 1: Initialize or adapt old centroids
#         # -----------------------------------------------------------
#         if x_syn_c is None:
#             # No old centroids, pick n_c random distinct points from new data
#             if x_c_torch.size(0) < n_c:
#                 raise ValueError(
#                     f"Not enough data to pick {n_c} distinct centroids for class {c}."
#                 )
#
#             rand_indices = torch.randperm(x_c_torch.size(0))[:n_c]
#             centers = x_c_torch[rand_indices].clone()
#         else:
#             # We do have some old centroids
#             old_count = x_syn_c.size(0)
#
#             if old_count == n_c:
#                 # Exactly match the number of desired clusters
#                 centers = x_syn_c.clone()
#             elif old_count < n_c:
#                 # Need more centroids => keep old + sample random new
#                 centers = x_syn_c.clone()
#                 needed = n_c - old_count
#                 if x_c_torch.size(0) < needed:
#                     raise ValueError(
#                         f"Not enough data to pick {needed} new centroids;"
#                         f" have only {x_c_torch.size(0)} points."
#                     )
#                 rand_indices = torch.randperm(x_c_torch.size(0))[:needed]
#                 new_centers = x_c_torch[rand_indices].clone()
#                 centers = torch.cat([centers, new_centers], dim=0)
#             else:
#                 # Too many old centroids => truncate
#                 centers = x_syn_c[:n_c].clone()
#
#         # Ensure we indeed have n_c centers
#         n_c = centers.size(0)
#
#         # -----------------------------------------------------------
#         # Step 2: Weighted K-Means or Fuzzy C-Means
#         # -----------------------------------------------------------
#         if args.fuzziness == 1.0:
#             # -----------------------
#             # Weighted K-Means branch
#             # -----------------------
#             if x_syn_c is not None:
#                 # 2.1) Old centroids => weighted points with sub-linear weighting
#                 x_syn_c = x_syn_c.to(device)
#                 if n_old is None:
#                     n_old = torch.ones(n_c // 4 * 3)
#                 n_old = n_old.to(device, dtype=torch.float32)
#
#                 # E.g. w_i = (n_old[i])^alpha
#                 w_old = n_old.pow(alpha)
#
#                 # Combine old centroids + new points
#                 # X_comb: [old_count + new_count, d]
#                 X_comb = torch.cat([x_syn_c, x_c_torch], dim=0)
#                 # W_comb: weights
#                 W_comb = torch.cat(
#                     [w_old, torch.ones(x_c_torch.size(0), device=device)], dim=0
#                 )
#
#                 # We'll do iterative Weighted K-Means with n_c clusters
#                 for iteration in range(max_iters):
#                     # Distances [points, n_c]
#                     diff = X_comb.unsqueeze(1) - centers.unsqueeze(
#                         0
#                     )  # [points, n_c, d]
#                     dist = (diff**2).sum(dim=2)  # [points, n_c]
#
#                     # Hard assignment => nearest center
#                     assignments = dist.argmin(dim=1)  # [points]
#
#                     # Update each centroid j => weighted mean of assigned points
#                     new_centers = centers.clone()
#                     for j in range(n_c):
#                         mask = assignments == j
#                         if mask.any():
#                             wsum = W_comb[mask].sum()
#                             wsum_points = (
#                                 X_comb[mask] * W_comb[mask].unsqueeze(1)
#                             ).sum(dim=0)
#                             new_centers[j] = wsum_points / (wsum + 1e-12)
#                         else:
#                             # If no points assigned, re-initialize randomly
#                             idx = torch.randint(0, X_comb.size(0), (1,)).item()
#                             new_centers[j] = X_comb[idx]
#
#                     shift = (new_centers - centers).pow(2).sum().sqrt().item()
#                     centers = new_centers
#                     if shift < epsilon:
#                         print(f"Weighted K-Means converged at iteration {iteration}")
#                         break
#                 else:
#                     print(
#                         f"Weighted K-Means reached {max_iters} iterations without convergence."
#                     )
#             else:
#                 # 2.2) No n_old => standard K-Means
#                 for iteration in range(max_iters):
#                     # Compute distances from points to centers
#                     diff = x_c_torch.unsqueeze(1) - centers.unsqueeze(
#                         0
#                     )  # [points, n_c, d]
#                     dist = (diff**2).sum(dim=2)  # [points, n_c]
#
#                     # Assignments: hard assignments to the nearest centroid
#                     assignments = dist.argmin(dim=1)  # [points]
#
#                     # Update centroids
#                     new_centers = centers.clone()
#                     for j in range(n_c):
#                         mask = assignments == j
#                         if mask.sum() > 0:
#                             new_centers[j] = x_c_torch[mask].mean(dim=0)
#                         else:
#                             # Handle empty clusters by reinitializing to a random point
#                             new_centers[j] = x_c_torch[
#                                 torch.randint(0, x_c_torch.size(0), (1,)).item()
#                             ]
#
#                     # Check for convergence
#                     center_shift = (new_centers - centers).pow(2).sum().sqrt().item()
#                     centers = new_centers
#                     if center_shift < epsilon:
#                         print(f"K-Means converged at iteration {iteration}")
#                         break
#                 else:
#                     print(
#                         f"K-Means reached maximum iterations ({max_iters}) without full convergence."
#                     )
#
#         return centers


def perform_balance_evolve_clusteringjk(
    x_c_current: np.ndarray,  # numpy array of shape (#points, d)
    y_syn,  # Not used in this snippet, but left for future expansions
    c,  # class label (unused in snippet, but you can log it if needed)
    x_syn_c,
    # previous centroids (Tensor or None)
    args,
    n_c: int,  # number of clusters for class c
    epsilon: float = 1e-5,  # Convergence threshold
    max_iters: int = 100,  # Maximum number of iterations
    save_p = False,
    class_id=0

) -> torch.Tensor:
    """
    Incremental Clustering with support for both K-Means and Fuzzy C-Means based on the fuzziness parameter.

    - x_c_current: Cumulative data for class c (numpy array of shape [#points, d])
    - y_syn: Not used in this snippet, but left for future expansions
    - c: Class label (unused in snippet, but can be used for logging)
    - x_syn_c: Previous centroids (Tensor or None). If provided but has a different number of centroids than n_c,
               it will be adjusted by adding or removing centroids.
    - args: Argument container with attributes 'device' and 'seed'.
    - data: Not used in this snippet
    - n_c: Number of clusters for class c
    - fuzziness: Fuzziness parameter m. If m=1.0, performs K-Means. If m > 1.0, performs Fuzzy C-Means.
    - epsilon: Convergence threshold for centroid/membership updates
    - max_iters: Maximum number of iterations to prevent infinite loops
    - returns: Updated centroids (Tensor of shape [n_c, d])
    """
    device = torch.device(args.device)
    x_c_torch = torch.tensor(x_c_current, dtype=torch.float32, device=device)

    # For reproducibility
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
         # ----- Initialization -----
        # For K-Means: work with torch tensors; for Fuzzy C-Means: work with numpy arrays.
        if args.fuzziness == 1.0:
            x_c_torch = torch.tensor(x_c_current, dtype=torch.float32, device=device)
            if x_syn_c is None:
                if x_c_torch.size(0) < n_c:
                    raise ValueError(f"Not enough data to pick {n_c} distinct centroids for class {c}.")
                rand_indices = torch.randperm(x_c_torch.size(0))[:n_c]
                centers = x_c_torch[rand_indices].clone()
            else:
                old_count = x_syn_c.shape[0]
                if old_count == n_c:
                    centers = x_syn_c.clone()
                elif old_count < n_c:
                    centers = x_syn_c.clone()
                    needed = n_c - old_count
                    if x_c_torch.size(0) < needed:
                        raise ValueError(f"Not enough data to add {needed} new centroids for class {c}.")
                    # Use your custom incremental_kmeanspp_init to choose new centroids
                    new_centers = incremental_kmeanspp_init(
                        new_data=x_c_torch,
                        old_centers=x_syn_c,
                        needed=needed,
                        device=device,
                    )
                    centers = torch.cat([centers, new_centers], dim=0)
                else:  # too many old centroids: truncate
                    centers = x_syn_c[:n_c].clone()

        else:  # Fuzzy C-Means branch: work in NumPy
            if x_syn_c is None:
                if x_c_current.shape[0] < n_c:
                    raise ValueError(f"Not enough data to pick {n_c} distinct centroids for class {c}.")
                rand_indices = random.permutation(x_c_current.shape[0])[:n_c]
                centers_np = x_c_current[rand_indices].copy()
            else:
                # Convert previous centroids to NumPy (if they aren't already)
                x_syn_np = x_syn_c.cpu().numpy() if isinstance(x_syn_c, torch.Tensor) else np.array(x_syn_c)
                old_count = x_syn_np.shape[0]
                if old_count == n_c:
                    centers_np = x_syn_np.copy()
                elif old_count < n_c:
                    centers_np = x_syn_np.copy()
                    needed = n_c - old_count
                    if x_c_current.shape[0] < needed:
                        raise ValueError(f"Not enough data to add {needed} new centroids for class {c}.")
                    rand_indices = random.permutation(x_c_current.shape[0])[:needed]
                    new_centers = x_c_current[rand_indices].copy()
                    centers_np = np.concatenate([centers_np, new_centers], axis=0)
                else:
                    centers_np = x_syn_np[:n_c].copy()

        # ----- Clustering -----
        if args.fuzziness == 1.0:
            # K-Means clustering (using Torch)
            rep_runs = args.rep_fuzz  # Use the same number of restarts as fuzzy C-means
            best_obj = float('inf')
            best_centers = None
            best_assignments = None

            for rep in range(rep_runs):
                current_seed = args.seed + rep
                with torch.random.fork_rng():
                    torch.manual_seed(current_seed)
                    
                    # Initialize centers for this run
                    if rep == 0 and centers is not None:
                        current_centers = centers.clone()
                    else:
                        # Random initialization
                        rand_indices = torch.randperm(x_c_torch.size(0))[:n_c]
                        current_centers = x_c_torch[rand_indices].clone()

                    # Run K-Means
                    for iteration in range(max_iters):
                        # Compute squared Euclidean distances: [points, n_c]
                        diff = x_c_torch.unsqueeze(1) - current_centers.unsqueeze(0)
                        dist = (diff ** 2).sum(dim=2)
                        # Hard assignment to the nearest center
                        current_assignments = dist.argmin(dim=1)
                        new_centers = current_centers.clone()
                        
                        # Update centers
                        for j in range(n_c):
                            mask = (current_assignments == j)
                            if mask.sum() > 0:
                                new_centers[j] = x_c_torch[mask].mean(dim=0)
                            else:
                                # Reinitialize an empty cluster to a random point
                                new_centers[j] = x_c_torch[torch.randint(0, x_c_torch.size(0), (1,)).item()]
                        
                        center_shift = (new_centers - current_centers).pow(2).sum().sqrt().item()
                        current_centers = new_centers
                        
                        if center_shift < epsilon:
                            break

                    # Compute balanced SSE objective
                    # 1. Compute base SSE
                    base_sse = dist.min(dim=1)[0].sum().item()
                    
                    # 2. Compute balance penalty
                    cluster_sizes = torch.bincount(current_assignments, minlength=n_c).float()
                    ideal_size = x_c_torch.size(0) / n_c
                    balance_penalty = ((cluster_sizes - ideal_size) ** 2).sum().item()
                    
                    # 3. Combine into final objective
                    print(f"base_sse: {base_sse}, balance_penalty: {balance_penalty}")
                    final_obj = base_sse + args.balance_alpha*balance_penalty

                    # Update best result if this run is better
                    if final_obj < best_obj:
                        best_obj = final_obj
                        best_centers = current_centers
                        best_assignments = current_assignments

            print(f"Best K-Means objective value: {best_obj}")
            centers = best_centers

            # ----- Compute Partition Matrix -----
            # Create a one-hot encoded partition matrix from the best assignments
            partition_matrix = torch.zeros((x_c_torch.size(0), n_c), device=x_c_torch.device)
            partition_matrix[torch.arange(x_c_torch.size(0)), best_assignments] = 1

            # Convert the partition matrix to a NumPy array
            partition_matrix_np = partition_matrix.cpu().numpy()

            # Save the partition matrix as a NumPy file if required
            if save_p:
                np.save(f"{args.dataset}_{args.si}_{class_id}_partition_matrix.npy", partition_matrix_np)

        else:
            # centers = parallel_fuzzy_cmeans(args, x_c_current, n_c=n_c, epsilon=epsilon, max_iters=max_iters, device=args.device, save_p=save_p,class_id=class_id)
            # Fuzzy C-Means clustering using skfuzzy
            m = args.fuzziness
            # Prepare data: skfuzzy expects data as (features, samples)
            data_T = np.array(x_c_current, dtype=np.float64).T  # shape: (d, n_samples)
            n_samples = data_T.shape[1]

            def init_membership_from_centers(centers_array, data):
                n_c_local = centers_array.shape[0]
                n_samples_local = data.shape[1]
                dists = pairwise_distances(data.T, centers_array)  # (n_samples, n_clusters)
                dists = np.clip(dists, 1e-10, None)
                power = np.clip(2 / (m - 1), -10, 10)
                inv_dists = np.clip(np.power(dists, -power), 0, 1e10)
                sum_inv_dists = np.sum(inv_dists, axis=0, keepdims=True)
                sum_inv_dists[sum_inv_dists == 0] = 1  # Avoid NaN
                U = inv_dists / sum_inv_dists  # Normalize
                return U.T  # Transpose to match expected output


            # If we have previous centroids (centers_np), compute initial membership; otherwise, random init.
            if 'centers_np' in locals():
                init_u = init_membership_from_centers(centers_np, data_T)
            else:
                init_u = np.random.rand(n_c, n_samples)
                init_u /= init_u.sum(axis=0, keepdims=True)

            # Use multiple restarts and choose the best clustering outcome.
            rep_runs = args.rep_fuzz
            best_obj = np.inf
            best_cntr = None

            for rep in range(rep_runs):
                current_seed = args.seed + rep
                # For the first run, if we computed an init membership, reuse it; else generate a random one.
                if rep == 0 and 'centers_np' in locals():
                    current_init_u = init_u.copy()
                else:
                    current_init_u = np.random.rand(n_c, n_samples)
                    current_init_u /= current_init_u.sum(axis=0, keepdims=True)

                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    data_T,
                    c=n_c,
                    m=m,
                    error=epsilon,
                    maxiter=max_iters,
                    init=current_init_u,
                    seed=current_seed
                )
                base_obj = jm[-1]
                # Compute effective cluster sizes: sum fuzzy memberships for each cluster.
                sizes = u.sum(axis=1)
                if isinstance(sizes, torch.Tensor):
                    sizes = sizes.detach().cpu().numpy()
                else:
                    sizes = np.asarray(sizes)
                ideal = float(n_samples) / float(n_c)
                balance_penalty = np.sum((sizes - ideal) ** 2)
                final_obj = base_obj + args.balance_alpha*balance_penalty

                if final_obj < best_obj:
                    best_obj = final_obj
                    best_cntr = cntr

            print("Best final objective value:", best_obj)
            centers = torch.from_numpy(best_cntr).float().to(device)
        return centers

def init_membership_from_centers(centers_array, data, m):
    n_samples_local = data.shape[1]
    dists = pairwise_distances(data.T, centers_array)  # (n_samples, n_clusters)
    dists = np.clip(dists, 1e-10, None)
    power = np.clip(2 / (m - 1), -10, 10)
    inv_dists = np.clip(np.power(dists, -power), 0, 1e10)
    sum_inv_dists = np.sum(inv_dists, axis=0, keepdims=True)
    sum_inv_dists[sum_inv_dists == 0] = 1  # Avoid NaN
    U = inv_dists / sum_inv_dists  # Normalize
    return U.T  # Transpose to match expected output

def run_fuzzy_cmeans(rep, args, data_T, n_c, m, epsilon, max_iters, init_u, n_samples):
    current_seed = args.seed + rep
    
    if rep == 0 and init_u is not None:
        current_init_u = init_u.copy()
    else:
        current_init_u = random.rand(n_c, n_samples)
        current_init_u /= current_init_u.sum(axis=0, keepdims=True)
    
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data_T,
        c=n_c,
        m=m,
        error=epsilon,
        maxiter=max_iters,
        init=current_init_u,
        seed=current_seed
    )
    
    base_obj = jm[-1]
    sizes = u.sum(axis=1)
    if isinstance(sizes, torch.Tensor):
        sizes = sizes.detach().cpu().numpy()
    else:
        sizes = np.asarray(sizes)
    ideal = float(n_samples) / float(n_c)
    balance_penalty = np.sum((sizes - ideal) ** 2)
    final_obj = base_obj + args.balance_alpha*balance_penalty
    
    return final_obj, cntr,u 

# def parallel_fuzzy_cmeans(args, x_c_current, n_c, epsilon, max_iters, device='cpu', save_p=False, class_id=0):
    m = args.fuzziness
    data_T = np.array(x_c_current, dtype=np.float64).T  # shape: (d, n_samples)
    n_samples = data_T.shape[1]

    if 'centers_np' in locals():
        init_u = init_membership_from_centers(centers_np, data_T, m)
    else:
        init_u = None

    rep_runs = args.rep_fuzz
    best_obj = float('inf')
    best_cntr = None
    best_partition_matrix = None  # Store the best U matrix

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_fuzzy_cmeans, rep, args, data_T, n_c, m, epsilon, max_iters, init_u, n_samples)
            for rep in range(rep_runs)
        ]

        for future in concurrent.futures.as_completed(futures):
            final_obj, cntr, U_matrix = future.result()  # Expecting U_matrix as an additional return value
            if final_obj < best_obj:
                best_obj = final_obj
                best_cntr = cntr
                best_partition_matrix = U_matrix  # Store the best partition matrix

    print("Best final objective value:", best_obj)

    updated_centroids_c = torch.from_numpy(best_cntr).float().to(device)

    # Save the best centroids and soft partition matrix if save_p is True
    if save_p:
        partition_save_path = f"{args.dataset}_{args.si}_{class_id}_partition_matrix.npy"
        np.save(partition_save_path, best_partition_matrix)
        print(f"Soft partition matrix saved to {partition_save_path}")

    return updated_centroids_c
def incremental_kmeanspp_init(
    new_data: torch.Tensor, old_centers: torch.Tensor, needed: int, device: torch.device
) -> torch.Tensor:
    """
    Pick `needed` new centroids from `new_data` via an incremental kmeans++ approach,
    given already-existing `old_centers`.
    """
    # If nothing to pick, return empty
    if needed <= 0:
        return torch.empty(0, old_centers.size(1), device=device)

    # We'll store chosen new centers here
    chosen = []

    # Step A: Precompute distance from each point to the nearest *old* center
    # ------------------------------------------------------------------------
    if old_centers is not None and old_centers.size(0) > 0:
        # dist_to_old: [N, old_count]
        diff_old = new_data.unsqueeze(1) - old_centers.unsqueeze(0)
        dist_to_old = (diff_old**2).sum(dim=2)  # [N, old_count]
        # nearest_dist: [N]
        nearest_dist = dist_to_old.min(dim=1).values
    else:
        # If no old centers, treat nearest_dist as infinity => pick first centroid random
        nearest_dist = torch.full((new_data.size(0),), float("inf"), device=device)

    # Step B: Iteratively pick `needed` new centroids
    # -----------------------------------------------
    for i in range(needed):
        # Probability of picking each point is proportional to nearest_dist^2
        dist_sq = nearest_dist  # shape [N]
        total = dist_sq.sum()

        if total <= 1e-12:
            # If sum of distances is extremely small, pick random to avoid numerical issues
            idx = torch.randint(0, new_data.size(0), (1,), device=device).item()
        else:
            # Sample a random threshold
            r = torch.rand(1, device=device).item() * total.item()

            # Walk through the cumulative sum to find which point crosses r
            cumulative = 0.0
            idx = 0
            for j in range(new_data.size(0)):
                cumulative += dist_sq[j].item()
                if cumulative >= r:
                    idx = j
                    break

        # Add the chosen center
        center_chosen = new_data[idx : idx + 1]  # shape [1, d]
        chosen.append(center_chosen)

        # Update nearest_dist to reflect new center
        diff_new = new_data - center_chosen  # shape [N, d]
        dist_new = (diff_new**2).sum(dim=1)  # shape [N]
        nearest_dist = torch.minimum(nearest_dist, dist_new)

    # Stack chosen new centroids
    new_centers = torch.cat(chosen, dim=0)
    return new_centers


# def perform_balance_evolve_clusteringjk(
#     x_c_current,  # numpy array of shape (#points, d)
#     y_syn,  # Not used in this snippet, but left for future expansions
#     c,  # class label (unused in snippet, but you can log it if needed)
#     n_c,  # desired number of clusters
#     x_syn_c,  # previous centroids (Tensor or None)
#     args,
#     data,
# ):
#     """
#     Incremental K-means with possible mismatch between old centroids and new desired cluster count.
#     - x_c_current: cumulative data for class c (numpy array)
#     - n_c: number of clusters for class c
#     - x_syn_c: previous centroids (None if first increment).
#       If x_syn_c is provided but has fewer/more rows than n_c, we handle it by adding or removing centroids.
#     - returns updated x_syn_c (shape: [n_c, d])
#     """
#     device = torch.device(args.device)
#     x_c_torch = torch.tensor(x_c_current, dtype=torch.float32, device=device)
#
#     # For reproducibility
#     with torch.random.fork_rng():
#         torch.manual_seed(args.seed)
#
#         # --- Step 1: Initialize or adapt old centroids ---
#         if x_syn_c is None:
#             # No old centroids: pick n_c random distinct points
#             if x_c_torch.size(0) < n_c:
#                 raise ValueError(
#                     f"Not enough data to pick {n_c} distinct centroids for class {c}."
#                 )
#             rand_indices = torch.randperm(x_c_torch.size(0))[:n_c]
#             centers = x_c_torch[rand_indices].clone()
#
#         else:
#             # Existing centroids
#             old_count = x_syn_c.shape[0]
#             if old_count == n_c:
#                 centers = x_syn_c.clone()
#             elif old_count < n_c:
#                 # Need more centroids: keep old, add random
#                 centers = x_syn_c.clone()
#                 needed = n_c - old_count
#                 if x_c_torch.size(0) < needed:
#                     raise ValueError(
#                         f"Not enough data to pick {needed} new centroids; have only {x_c_torch.size(0)} points."
#                     )
#                 rand_indices = torch.randperm(x_c_torch.size(0))[:needed]
#                 new_centers = x_c_torch[rand_indices].clone()
#                 centers = torch.cat([centers, new_centers], dim=0)
#             else:
#                 # Too many old centroids: reduce them
#                 centers = x_syn_c[:n_c].clone()
#
#         # --- Step 2: Standard K-means ---
#         num_iters = 100
#         for _ in range(num_iters):
#             # Compute distance from each point to each center
#             diff = x_c_torch.unsqueeze(1) - centers.unsqueeze(0)  # [points, n_c, d]
#             dist = (diff**2).sum(dim=2)  # [points, n_c]
#
#             assignments = dist.argmin(dim=1)  # [points]
#
#             # Update each cluster center
#             for j in range(n_c):
#                 mask = assignments == j
#                 if mask.sum() > 0:
#                     centers[j] = x_c_torch[mask].mean(dim=0)
#
#         return centers
