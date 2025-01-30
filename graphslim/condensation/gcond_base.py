from collections import Counter

import torch.nn as nn

from graphslim.coarsening import *
from graphslim.condensation.utils import *
from graphslim.models import *
from graphslim.sparsification import *
from graphslim.utils import *
from graphslim.dataset.utils import save_reduced


class GCondBase:
    """
    A base class for graph condition generation and training.

    Parameters
    ----------
    setting : str
        The setting for the graph condensation process.
    data : object
        The data object containing the dataset.
    args : Namespace
        Arguments and hyperparameters for the model and training process.
    **kwargs : keyword arguments
        Additional arguments for initialization.
    """

    def __init__(self, setting, data, args, **kwargs):
        """
        Initializes a GCondBase instance.

        Parameters
        ----------
        setting : str
            The type of experimental setting.
        data : object
            The graph data object, which includes features, adjacency matrix, labels, etc.
        args : Namespace
            Arguments object containing hyperparameters for training and model.
        **kwargs : keyword arguments
            Additional optional parameters.
        """
        self.data = data
        self.args = args
        self.device = args.device
        self.setting = setting

        if args.method not in ['msgc']:
            self.labels_syn = self.data.labels_syn = self.generate_labels_syn(data)
            n = self.nnodes_syn = self.data.labels_syn.shape[0]
        else:
            n = self.nnodes_syn = int(data.feat_train.shape[0] * args.reduction_rate)
        self.d = d = data.feat_train.shape[1]
        # self.d = d = 64
        print(f'target reduced size:{int(data.feat_train.shape[0] * args.reduction_rate)}')
        print(f'actual reduced size:{n}')

        # from collections import Counter; print(Counter(data.labels_train))

        self.feat_syn = nn.Parameter(torch.empty(n, d).to(self.device))
        if args.method not in ['sgdd', 'gcsntk', 'msgc']:
            self.pge = PGE(nfeat=d, nnodes=n, device=self.device, args=args).to(self.device)
            self.adj_syn = None

            # self.reset_parameters()
            self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
            self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
            print('adj_syn:', (n, n), 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        """
        Resets the parameters of the model.
        """
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))
        self.pge.reset_parameters()

    def generate_labels_syn(self, data):
        """
        Generates synthetic labels to match the target number of samples.

        Parameters
        ----------
        data : object
            The graph data object, which includes features, adjacency matrix, labels, etc.

        Returns
        -------
        np.ndarray
            A numpy array of synthetic labels.
        """
        counter = Counter(data.labels_train.tolist())
        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                # only clip labels with largest number of samples
                num_class_dict[c] = max(int(n * self.args.reduction_rate) - sum_, 1)
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
        self.data.num_class_dict = self.num_class_dict = num_class_dict
        if self.args.verbose:
            print(num_class_dict)
        return np.array(labels_syn)

    def init(self, with_adj=False, reuse_init=False):
        """
        Initializes synthetic features and (optionally) adjacency matrix.

        Parameters
        ----------
        with_adj : bool, optional
            Whether to initialize the adjacency matrix (default is False).

        Returns
        -------
        tuple
            A tuple containing the synthetic features and (optionally) the adjacency matrix.
        """
        args = self.args
        if args.init == 'clustering':
            if args.agg:
                agent = ClusterAgg(setting=args.setting, data=self.data, args=args)
            else:
                agent = Cluster(setting=args.setting, data=self.data, args=args)
        elif args.init == 'averaging':
            agent = Average(setting=args.setting, data=self.data, args=args)
        elif args.init == 'kcenter':
            agent = KCenter(setting=args.setting, data=self.data, args=args)
        elif args.init == 'herding':
            agent = Herding(setting=args.setting, data=self.data, args=args)
        elif args.init == 'cent_p':
            agent = CentP(setting=args.setting, data=self.data, args=args)
        elif args.init == 'cent_d':
            agent = CentD(setting=args.setting, data=self.data, args=args)
        else:
            agent = Random(setting=args.setting, data=self.data, args=args)

        if reuse_init:
            save_path = f'{args.save_path}/reduced_graph/{args.init}'
            if with_adj and os.path.exists(f'{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt'):
                feat_syn = torch.load(
                        f'{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location=args.device)
                return feat_syn, adj_syn
            if os.path.exists(f'{save_path}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt'):
                feat_syn = torch.load(
                        f'{save_path}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location=args.device)
                return feat_syn
        temp = args.method
        args.method = args.init
        reduced_data = agent.reduce(self.data, verbose=True, save=True)
        args.method = temp
        if with_adj:
            return reduced_data.feat_syn, reduced_data.adj_syn
        else:
            return reduced_data.feat_syn
            #return reduced_data.feat_syn_0, reduced_data.feat_syn_1,reduced_data.feat_syn_2



    def train_class(self, model, adj, features, labels, labels_syn, args, soft=False):
        """
        Trains the model and computes the loss.

        Parameters
        ----------
        model : torch.nn.Module
            The model object.
        adj : torch.Tensor
            The adjacency matrix.
        features : torch.Tensor
            The feature matrix.
        labels : torch.Tensor
            The actual labels.
        labels_syn : torch.Tensor
            The synthetic labels.
        args : Namespace
            Arguments object containing hyperparameters for training and model.

        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        data = self.data
        feat_syn = self.feat_syn
        adj_syn = self.adj_syn
        loss = torch.tensor(0.0, device=self.device)

        if not soft:
            loss_fn = F.nll_loss
            # Convert labels to class indices if they are one-hot encoded
            if labels.dim() > 1:
                hard_labels = torch.argmax(labels, dim=-1)
            else:
                hard_labels = labels.long()
            if labels_syn.dim() > 1:
                hard_labels_syn = torch.argmax(labels_syn, dim=-1)
            else:
                hard_labels_syn = labels_syn.long()
        else:
            loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
            # Convert labels to one-hot encoding if they are class indices
            if labels.dim() == 1:
                hard_labels = labels
                soft_labels = F.one_hot(labels, num_classes=data.nclass).float()
            if labels_syn.dim() == 1:
                hard_labels_syn = labels
                soft_labels_syn = F.one_hot(labels_syn, num_classes=data.nclass).float()
            else:
                hard_labels_syn = torch.argmax(labels_syn, dim=-1)
                soft_labels_syn = labels_syn

        # Loop over each class
        for c in range(data.nclass):
            # Retrieve a batch of real data samples for class c
            batch_size, n_id, adjs = data.retrieve_class_sampler(c, adj, args)
            adjs = [adj[0].to(self.device) for adj in adjs]
            input_real = features[n_id].to(self.device)
            if soft:
                labels_real = soft_labels[n_id[:batch_size]].to(self.device)
            else:
                labels_real = hard_labels[n_id[:batch_size]].to(self.device)

            output_real = model(input_real, adjs)
            loss_real = loss_fn(output_real, labels_real)

            gw_real = torch.autograd.grad(loss_real, model.parameters(), retain_graph=True)
            gw_real = [g.detach().clone() for g in gw_real]

            output_syn = model(feat_syn, adj_syn)
            if soft:
                loss_syn = loss_fn(output_syn[hard_labels_syn == c], soft_labels_syn[hard_labels_syn == c])
            else:
                loss_syn = loss_fn(output_syn[hard_labels_syn == c], hard_labels_syn[hard_labels_syn == c])


            # Compute gradients w.r.t. model parameters for synthetic data
            gw_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

            # Compute matching loss between gradients
            coeff = self.num_class_dict[c] / self.nnodes_syn
            ml = match_loss(gw_syn, gw_real, args, device=self.device)
            loss += coeff * ml

        return loss

    def get_loops(self, args):
        # Get the two hyper-parameters of outer-loop and inner-loop.
        # The following values are empirically good.
        """
        Retrieves the outer-loop and inner-loop hyperparameters.

        Parameters
        ----------
        args : Namespace
            Arguments object containing hyperparameters for training and model.

        Returns
        -------
        tuple
            Outer-loop and inner-loop hyperparameters.
        """
        return args.outer_loop, args.inner_loop

    def check_bn(self, model):
        """
        Checks if the model contains BatchNorm layers and fixes their mean and variance after training.

        Parameters
        ----------
        model : torch.nn.Module
            The model object.

        Returns
        -------
        torch.nn.Module
            The model with BatchNorm layers fixed.
        """
        BN_flag = False
        for module in model.modules():
            if 'BatchNorm' in module._get_name():  # BatchNorm
                BN_flag = True
        if BN_flag:
            model.train()  # for updating the mu, sigma of BatchNorm
            # output_real = model.forward(features, adj)
            for module in model.modules():
                if 'BatchNorm' in module._get_name():  # BatchNorm
                    module.eval()  # fix mu and sigma of every BatchNorm layer
        return model

    def intermediate_evaluation(self, best_val, loss_avg=None, save=True, save_valid_acc=False):
        """
        Performs intermediate evaluation and saves the best model.

        Parameters
        ----------
        best_val : float
            The best validation accuracy observed so far.
        loss_avg : float
            The average loss.
        save : bool, optional
            Whether to save the model (default is True).

        Returns
        -------
        float
            The updated best validation accuracy.
        """
        data = self.data
        args = self.args
        if args.verbose:
            print('loss_avg: {}'.format(loss_avg))

        res = []

        for i in range(args.run_inter_eval):
            res.append(
                self.test_with_val(verbose=False, setting=args.setting, iters=args.eval_epochs))

        res = np.array(res).T
        current_val = res[0].mean()
        args.logger.info('\nVal:  {:.4f} +/- {:.4f}'.format(100*current_val, 100*res[0].std()))
        args.logger.info('Test: {:.4f} +/- {:.4f}'.format(100*res[1].mean(), 100*res[1].std()))

        if save and current_val > best_val:
            best_val = current_val
            save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)
        return best_val

    def test_with_val(self, verbose=False, setting='trans', iters=200, best_val=None):
        """
        Conducts validation testing and returns results.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose output (default is False).
        setting : str, optional
            The setting type (default is 'trans').
        iters : int, optional
            Number of iterations for validation testing (default is 200).

        Returns
        -------
        list
            A list containing validation results.
        """

        args, data, device = self.args, self.data, self.device

        model = eval(args.final_eval_model)(data.feat_syn.shape[1], args.hidden, data.nclass, args, mode='eval').to(device)

        acc_val = model.fit_with_val(data,
                                     train_iters=iters, normadj=True, verbose=False,
                                     setting=setting, reduced=True, best_val=best_val)

        model.eval()
        acc_test = model.test(data, setting=setting,verbose=False)
        # if verbose:
        #     print('Val Accuracy and Std:',
        #           repr([res.mean(0), res.std(0)]))
        return [acc_val, acc_test]
