import torch

from graphslim.coarsening.coarsening_base import Coarsen
from graphslim.coarsening.labels import synthetic_labels, label_tensor
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory


class Average(Coarsen):
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
        super(Average, self).__init__(setting, data, args)
        args.method = "averaging"

    @verbose_time_memory
    def reduce(self, data, verbose=True, save=True):
        """
        Reduces the data by averaging features for each class.

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
        if getattr(data, "labels_syn", None) is not None:
            y_syn = data.labels_syn
            self.labels_train = data.labels_train
            y_train = data.labels_train
            x_train = data.feat_train
        else:
            y_syn, y_train, x_train = self.prepare_select(data, args)

        y_syn = label_tensor(y_syn, x_train)
        x_syn = x_train.new_zeros(y_syn.shape[0], x_train.shape[1])
        for c in range(n_classes):
            x_c = x_train[y_train == c]
            x_syn[y_syn == c] = x_c.mean(0)

        data.feat_syn, data.labels_syn = x_syn.to(x_train.device), y_syn
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
