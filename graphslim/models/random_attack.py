import numpy as np
from deeprobust.graph.global_attack import BaseAttack
import scipy.sparse as sp
import torch


# import random


class RandomAttack(BaseAttack):
    """Randomly adding noise to the input graph

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.global_attack import Random
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> model = Random()
    >>> model.attack(adj, n_perturbations=10)
    >>> modified_adj = model.modified_adj

    """

    def __init__(
        self, model=None, nnodes=None, attack_structure=True, attack_features=False, device="cpu"
    ):
        """Initialize RandomAttack and its configuration.

        :param model: Model instance used for training, prediction, or cache attachment. Defaults to ``None``.
        :param nnodes: Number of nodes used to construct candidate edge pairs. Defaults to ``None``.
        :param attack_structure: Whether the random attacker may modify graph edges. Defaults to ``True``.
        :param attack_features: Whether feature attacks are enabled; unsupported modes are rejected by the attacker. Defaults to ``False``.
        :param device: Target PyTorch device, such as ``cpu`` or ``cuda:0``. Defaults to ``'cpu'``.
        """
        super(RandomAttack, self).__init__(
            model,
            nnodes,
            attack_structure=attack_structure,
            attack_features=attack_features,
            device=device,
        )

    def attack(self, target, n_perturbations, type="add", **kwargs):
        """Generate attacks on the input graph.

        Parameters
        ----------
        target : scipy.sparse.csr_matrix or torch.Tensor
            Original adjacency when attacking structure, or original feature
            tensor when attacking features.
        n_perturbations : int
            Number of edge modifications or feature columns to zero.
        type : str, optional
            Edge perturbation: 'add', 'remove' or 'flip'. Defaults to 'add'.
        **kwargs : dict
            Compatibility keyword arguments; currently unused.

        Returns
        -------
        None
            Results are stored in ``modified_adj`` and/or ``modified_features``.
        """

        if self.attack_structure:
            modified_adj = self.perturb_adj(target, n_perturbations, type)
            self.modified_adj = modified_adj
        if self.attack_features:
            self.modified_features = self.perturb_features(target, n_perturbations)

    def perturb_adj(self, adj, n_perturbations, type="add"):
        """Randomly add, remove or flip edges.

        Parameters
        ----------
        adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of edge removals/additions.
        type: str
            perturbation type. Could be 'add', 'remove' or 'flip'.

        Returns
        ------
        scipy.sparse matrix
            perturbed adjacency matrix
        """
        # adj: sp.csr_matrix
        modified_adj = adj.tolil()

        type = type.lower()
        assert type in ["add", "remove", "flip"]

        if type == "flip":
            # sample edges to flip
            edges = self.random_sample_edges(adj, n_perturbations, exclude=set())
            for n1, n2 in edges:
                modified_adj[n1, n2] = 1 - modified_adj[n1, n2]
                modified_adj[n2, n1] = 1 - modified_adj[n2, n1]

        if type == "add":
            # sample edges to add
            nonzero = set(zip(*adj.nonzero()))
            edges = self.random_sample_edges(adj, n_perturbations, exclude=nonzero)
            for n1, n2 in edges:
                modified_adj[n1, n2] = 1
                modified_adj[n2, n1] = 1

        if type == "remove":
            # sample edges to remove
            nonzero = np.array(sp.triu(adj, k=1).nonzero()).T
            indices = np.random.permutation(nonzero)[:n_perturbations].T
            modified_adj[indices[0], indices[1]] = 0
            modified_adj[indices[1], indices[0]] = 0

        self.check_adj(modified_adj)
        return modified_adj

    def perturb_features(self, features, n_perturbations):
        """Randomly perturb features by setting n_perturbations columns to zero.

        Args:
            features (tensor): The tensor of features to perturb.
            n_perturbations (int): The number of columns to set to zero.

        Returns:
            tensor: The perturbed feature tensor.
        """

        # Create a copy of the features tensor to modify
        modified_features = features.clone()

        # Randomly choose column indices to perturb
        columns_to_perturb = torch.randperm(features.size(1))[:n_perturbations]

        # Set the chosen columns to zero
        modified_features[:, columns_to_perturb] = 0

        return modified_features

    def inject_nodes(self, adj, n_add, n_perturbations):
        """Unsupported node-injection attack entry point.

        Always raises ``NotImplementedError``.

        :param adj: Graph adjacency; rows and columns refer to the same node ordering as the features.
        :param n_add: Requested number of injected nodes; node injection is currently unsupported.
        :param n_perturbations: Maximum number of edge flips allowed by the attack budget.
        """
        # adj: sp.csr_matrix
        print("number of pertubations: %s" % n_perturbations)
        raise NotImplementedError

        modified_adj = adj.tolil()
        return modified_adj

    def random_sample_edges(self, adj, n, exclude):
        """Draw a fixed number of allowed random edge pairs.

        :param adj: Graph adjacency; rows and columns refer to the same node ordering as the features.
        :param n: Number of edge pairs to sample.
        :param exclude: Set of forbidden endpoint pairs; new pairs are added to this set.
        :returns: List of sampled edge endpoint pairs.
        """
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

    def sample_forever(self, adj, exclude):
        """Sample unique non-diagonal node pairs indefinitely while updating the exclusion set.

        :param adj: Graph adjacency; rows and columns refer to the same node ordering as the features.
        :param exclude: Set of forbidden endpoint pairs; new pairs are added to this set.
        :returns: Iterator of allowed edge endpoint pairs.
        """
        while True:
            # t = tuple(np.random.randint(0, adj.shape[0], 2))
            # t = tuple(random.sample(range(0, adj.shape[0]), 2))
            t = tuple(np.random.choice(adj.shape[0], 2, replace=False))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))
