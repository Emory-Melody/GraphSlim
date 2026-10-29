import torch
from sklearn.cluster import BisectingKMeans
from torch_scatter import scatter_mean

from graphslim.coarsening.coarsening_base import Coarsen
from graphslim.coarsening.labels import synthetic_labels, label_tensor
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.reduction.cluster_ops import fitted_labels


class Cluster(Coarsen):
    """
    A structure-free coarsening method that also serves as initialization for condensation methods.
    Outputs synthesized features (`feat_syn`) and labels (`label_syn`).

    Parameters
    ----------
    setting : str
        Configuration setting.
    data : object
        Data object containing the graph and feature information.
    args : object
        Arguments containing various settings for the coarsening process.
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(self, setting, data, args):
        super(Cluster, self).__init__(setting, data, args)
        args.method = "clustering"

    @verbose_time_memory
    def reduce(self, data, verbose=True, save=True):
        """
        Reduces the data by clustering features for each class using Bisecting K-means.

        Parameters
        ----------
        data : object
            The data to be reduced.
        verbose : bool, optional
            If True, prints verbose output. Defaults to True.
        save : bool, optional
            If True, saves the reduced data. Defaults to True.

        Returns
        -------
        object
            The reduced data with synthesized features and labels.
        """
        args = self.args
        n_classes = data.nclass
        if hasattr(data, "labels_syn") and data.labels_syn is not None:
            y_syn = data.labels_syn
            self.labels_train = data.labels_train
            y_train = data.labels_train
            x_train = data.feat_train
        else:
            y_syn, y_train, x_train = self.prepare_select(data, args)

        y_syn = label_tensor(y_syn, x_train).cpu()
        x_syn = torch.zeros(y_syn.shape[0], x_train.shape[1], dtype=x_train.dtype)
        for c in range(n_classes):
            x_c = x_train[y_train == c].cpu()
            n_c = (y_syn == c).sum().item()
            if n_c == 0:
                continue
            k_means = BisectingKMeans(n_clusters=n_c, random_state=0)
            clusters = torch.from_numpy(fitted_labels(k_means, x_c)).long()
            x_syn[y_syn == c] = scatter_mean(x_c, clusters, dim=0, dim_size=n_c)

        data.feat_syn, data.labels_syn = x_syn.to(x_train.device), y_syn.to(x_train.device)
        data.adj_syn = torch.eye(data.feat_syn.shape[0])

        if save:
            save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)

        return data

    def prepare_select(self, data, args):
        """
        Prepares and selects synthetic labels and features for coarsening.

        Parameters
        ----------
        data : object
            The data to be processed.
        args : object
            Arguments containing various settings for the coarsening process.

        Returns
        -------
        tuple
            Synthetic labels, training labels and training features, in that order.
        """
        feat_train = data.feat_train
        labels_train = data.labels_train

        labels_syn = synthetic_labels(labels_train, args.reduction_rate)

        return labels_syn, labels_train, feat_train
