import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange, tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_sparse import matmul

from torch_geometric.utils import dense_to_sparse
from graphslim.dataset import *
from graphslim.evaluation import *
from graphslim.evaluation.utils import *
from graphslim.models import *
from torch_sparse import SparseTensor
from graphslim.dataset.convertor import ei2csr
from graphslim.utils import accuracy, seed_everything, normalize_adj_tensor, to_tensor, is_sparse_tensor, is_identity, \
    f1_macro


class Evaluator:
    """
    A class to evaluate different models and their hyperparameters on graph data.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments and configuration parameters.
    **kwargs : keyword arguments
        Additional parameters.
    """

    def __init__(self, args, **kwargs):
        """
        Initializes the Evaluator with given arguments.

        Parameters
        ----------
        args : argparse.Namespace
            Command-line arguments and configuration parameters.
        **kwargs : keyword arguments
            Additional parameters.
        """
        self.args = args
        self.device = args.device
        self.reset_parameters()
        self.metric = accuracy if args.metric == 'accuracy' else f1_macro

    def reset_parameters(self):
        """
        Initializes or resets model parameters.
        """
        pass

    def grid_search(self, data, model_type, param_grid, reduced=True):
        """
        Performs a grid search over hyperparameters.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        model_type : str
            The type of model used for evaluation.
        param_grid : dict
            A dictionary containing parameter grids for grid search.
        reduced : bool, optional, default=True
            Whether to use synthetic data.

        Returns
        -------
        best_test_result : tuple
            Best test result as (mean_accuracy, std_accuracy).
        best_params : dict
            Best parameters found during grid search.
        """
        args = self.args
        best_val_result = None
        best_test_result = None
        best_params = None

        for params in tqdm(ParameterGrid(param_grid)):
            for key, value in params.items():
                setattr(args, key, value)

            res = []
            for i in range(args.run_eval):
                seed_everything(i)
                res.append([self.test(data, model_type=model_type, verbose=False, reduced=reduced, mode='cross')])
                torch.cuda.empty_cache()

            res = np.array(res).reshape(args.run_eval, -1)
            res_mean, res_std = res.mean(axis=0), res.std(axis=0)

            if args.verbose:
                print(
                    f'{model_type} Test results with params {params}: {100 * res_mean[1]:.2f} +/- {100 * res_std[1]:.2f}')

            if best_val_result is None or res_mean[0] > best_val_result[0]:
                best_val_result = (res_mean[0], res_std[0])
                best_test_result = (res_mean[1], res_std[1])
                best_params = params

        return best_test_result, best_params

    def train_cross(self, data, grid_search=True, reduced=True):
        """
        Trains models and performs grid search if required.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        grid_search : bool, optional, default=True
            Whether to perform grid search over hyperparameters.
        reduced : bool, optional, default=True
            Whether to use synthetic data.
        """
        args = self.args

        if grid_search:
            gs_params = {
                'GCN': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                        'dropout': [0.0, 0.5]},
                'SGC': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                        'dropout': [0.0, 0.5], 'ntrans': [1, 2]},
                'APPNP': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                          'dropout': [0.0, 0.5], 'ntrans': [1, 2], 'alpha': [0.1, 0.2]},
                'Cheby': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                          'dropout': [0.0, 0.5]},
                'GraphSage': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                              'dropout': [0.0, 0.5]},
                'GAT': {'hidden': [16, 64], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                        'dropout': [0.0, 0.5, 0.7]},
                'SGFormer': {'trans_num_layers': [1, 2, 3], 'lr': [0.01, 0.001], 'trans_weight_decay': [0.001, 0.0001],
                             'trans_dropout': [0.0, 0.5, 0.7]}
            }

            if args.dataset in ['reddit']:
                gs_params['GAT']['hidden'] = [8, 16]

            for model_type in gs_params:
                if reduced:
                    data.feat_syn, data.adj_syn, data.labels_syn = get_syn_data(data, model_type=model_type,
                                                                                verbose=args.verbose)
                print(f'Starting Grid Search for {model_type}')
                best_result, best_params = self.grid_search(data, model_type, gs_params[model_type], reduced=reduced)
                args.logger.info(
                    f'Best {model_type} Test Result: {100 * best_result[0]:.2f} +/- {100 * best_result[1]:.2f} with params {best_params}')
        else:
            eval_model_list = ['GCN', 'SGC', 'APPNP', 'Cheby', 'GraphSage', 'GAT']
            for model_type in eval_model_list:
                data.feat_syn, data.adj_syn, data.labels_syn = get_syn_data(data, model_type=model_type,
                                                                            verbose=args.verbose)
                best_result = self.evaluate(data, model_type=args.eval_model)
                args.logger.info(
                    f'{model_type} Result: {100 * best_result[0]:.2f} +/- {100 * best_result[1]:.2f}')

    def test(self, data, model_type, verbose=True, reduced=True, mode='eval', MIA=False):
        """
        Tests a model and returns accuracy and loss.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        model_type : str
            The type of model to test.
        verbose : bool, optional, default=True
            Whether to print detailed logs.
        reduced : bool, optional, default=True
            Whether to use synthetic data.
        mode : str, optional, default='eval'
            The mode for the model (e.g., 'eval' or 'cross').

        Returns
        -------
        best_acc_val : float
            Best accuracy on validation set.
        acc_test : float
            Accuracy on test set.
        """
        args = self.args

        if verbose:
            print(f'======= testing {model_type}')

        model = eval(model_type)(data.feat_full.shape[1], args.hidden, data.nclass, args, mode=mode).to(self.device)
        best_acc_val = model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True, verbose=verbose,
                                          setting=args.setting, reduced=reduced)

        model.eval()
        labels_test = data.labels_test.long().to(args.device)
        labels_train = data.labels_train.long().to(args.device)

        if args.attack is not None:
            data = attack(data, args)

        if args.setting == 'ind':
            output = model.predict(data.feat_test, data.adj_test)
            loss_test = F.nll_loss(output, labels_test)
            acc_test = self.metric(output, labels_test)
            if MIA:
                output_train = model.predict(data.feat_train, data.adj_train)
                conf_train = F.softmax(output_train, dim=1)
                conf_test = F.softmax(output, dim=1)

            if verbose:
                print("Test set results:",
                      f"loss= {loss_test.item():.4f}",
                      f"accuracy= {acc_test.item():.4f}")
        else:
            output = model.predict(data.feat_full, data.adj_full)
            loss_test = F.nll_loss(output[data.idx_test], labels_test)
            acc_test = self.metric(output[data.idx_test], labels_test)
            if MIA:
                conf_train = F.softmax(output[data.idx_train], dim=1)
                conf_test = F.softmax(output[data.idx_test], dim=1)
            if verbose:
                print("Test full set results:",
                      f"loss= {loss_test.item():.4f}",
                      f"accuracy= {acc_test.item():.4f}")
        if MIA:
            mia_acc = inference_via_confidence(conf_train.cpu().numpy(), conf_test.cpu().numpy(), labels_train.cpu(),
                                               labels_test.cpu())
            # print(f"MIA accuracy: {mia_acc}")
            return best_acc_val.item(), acc_test.item(), mia_acc
        return best_acc_val.item(), acc_test.item()

    def evaluate(self, data, model_type, verbose=True, reduced=True, mode='eval'):
        """
        Evaluates a model over multiple runs and returns mean and standard deviation of accuracy.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        model_type : str
            The type of model to evaluate.
        verbose : bool, optional, default=True
            Whether to print detailed logs.
        reduced : bool, optional, default=True
            Whether to use synthetic data.
        mode : str, optional, default='eval'
            The mode for the model (e.g., 'eval' or 'cross').

        Returns
        -------
        mean_acc : float
            Mean accuracy over multiple runs.
        std_acc : float
            Standard deviation of accuracy over multiple runs.
        """

        args = self.args

        # Prepare synthetic data if required
        if reduced:
            data.feat_syn, data.adj_syn, data.labels_syn = get_syn_data(data, args, model_type=model_type,
                                                                        verbose=verbose)

        # Initialize progress bar based on verbosity
        if verbose:
            print(f'Evaluating reduced data using {model_type}')
            run_evaluation = trange(args.run_eval)
        else:
            run_evaluation = range(args.run_eval)

        # Collect accuracy results from multiple runs
        res = []
        for i in run_evaluation:
            seed_everything(args.seed + i)
            _, best_acc = self.test(data, model_type=model_type, verbose=args.verbose, reduced=reduced,
                                    mode=mode)
            res.append(best_acc)
            if verbose:
                run_evaluation.set_postfix(test_acc=best_acc)

        res = np.array(res)

        # Log and return mean and standard deviation of accuracy
        args.logger.info(f'Seed:{args.seed}, Test Mean Accuracy: {100 * res.mean():.2f} +/- {100 * res.std():.2f}')
        return res.mean(), res.std()

    def MIA_evaluate(self, data, model_type, verbose=True, reduced=True, mode='eval'):
        """
        Evaluates a model over multiple runs and returns mean and standard deviation of accuracy.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        model_type : str
            The type of model to evaluate.
        verbose : bool, optional, default=True
            Whether to print detailed logs.
        reduced : bool, optional, default=True
            Whether to use synthetic data.
        mode : str, optional, default='eval'
            The mode for the model (e.g., 'eval' or 'cross').

        Returns
        -------
        mean_acc : float
            Mean accuracy over multiple runs.
        std_acc : float
            Standard deviation of accuracy over multiple runs.
        """

        args = self.args

        # Prepare synthetic data if required
        if reduced:
            data.feat_syn, data.adj_syn, data.labels_syn = get_syn_data(data, args, model_type=model_type,
                                                                        verbose=verbose)

        # Initialize progress bar based on verbosity
        if verbose:
            print(f'Evaluating reduced data using {model_type}')
            run_evaluation = trange(args.run_eval)
        else:
            run_evaluation = range(args.run_eval)

        # Collect accuracy results from multiple runs
        res = []
        mia_res = []
        for i in run_evaluation:
            seed_everything(args.seed + i)
            _, best_acc, mia_acc = self.test(data, model_type=model_type, verbose=args.verbose, reduced=reduced,
                                             mode=mode, MIA=True)
            res.append(best_acc)
            mia_res.append(mia_acc)
            if verbose:
                run_evaluation.set_postfix(test_acc=best_acc,MIA_acc=mia_acc)

        res = np.array(res)
        mia_res = np.array(mia_res)

        # Log and return mean and standard deviation of accuracy
        args.logger.info(f'Seed:{args.seed}, Test Mean Accuracy: {100 * res.mean():.2f} +/- {100 * res.std():.2f}, '
                         f'MIA Accuracy: {100 * mia_res.mean():.2f} +/- {100 * mia_res.std():.2f}')
        return res.mean(), res.std()

    def nas_evaluate(self, data, model_type, verbose=False, reduced=None):
        """
        Evaluates a model for neural architecture search (NAS) and returns mean and standard deviation of validation accuracy.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        model_type : str
            The type of model to evaluate.
        verbose : bool, optional, default=False
            Whether to print detailed logs.
        reduced : bool, optional, default=None
            Whether to use synthetic data.

        Returns
        -------
        mean_acc_val : float
            Mean validation accuracy over multiple runs.
        std_acc_val : float
            Standard deviation of validation accuracy over multiple runs.
        """
        args = self.args
        res = []

        # Prepare synthetic data if required
        data.feat_syn, data.adj_syn, data.labels_syn = get_syn_data(data, model_type=model_type, verbose=verbose)

        # Initialize progress bar based on verbosity
        if verbose:
            run_evaluation = trange(args.run_evaluation)
        else:
            run_evaluation = range(args.run_evaluation)

        # Collect validation accuracy results from multiple runs
        for i in run_evaluation:
            model = eval(model_type)(data.feat_syn.shape[1], args.hidden, data.nclass, args, mode='eval').to(
                self.device)
            best_acc_val = model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True, verbose=verbose,
                                              setting=args.setting, reduced=reduced)
            res.append(best_acc_val)
            if verbose:
                run_evaluation.set_postfix(best_acc_val=best_acc_val.item())

        res = np.array(res)

        # Print and return mean and standard deviation of validation accuracy
        if verbose:
            print(f'Validation Mean Accuracy: {100 * res.mean():.2f} +/- {100 * res.std():.2f}')

        return res.mean(), res.std()

    def tsne_vis(self, feat_train, labels_train, feat_syn, labels_syn):
        """
        Visualize t-SNE for original and synthetic data.

        Parameters:
            feat_train (torch.tensor): Original features.
            labels_train (torch.tensor): Labels for original features.
            feat_syn (torch.tensor): Synthetic features.
            labels_syn (torch.tensor): Labels for synthetic features.
        """
        labels_train_np = labels_train.cpu().numpy()
        feat_train_np = feat_train.cpu().numpy()
        labels_syn_np = labels_syn.cpu().numpy()
        feat_syn_np = feat_syn.cpu().numpy()

        # Separate features based on labels for original and synthetic data
        data_feat_ori_0 = feat_train_np[labels_train_np == 0]
        data_feat_ori_1 = feat_train_np[labels_train_np == 1]
        data_feat_syn_0 = feat_syn_np[labels_syn_np == 0]
        data_feat_syn_1 = feat_syn_np[labels_syn_np == 1]

        # Concatenate all features for t-SNE visualization
        all_data = np.concatenate((data_feat_ori_0, data_feat_ori_1, data_feat_syn_0, data_feat_syn_1), axis=0)
        perplexity_value = min(30, len(all_data) - 1)

        # Apply t-SNE to reduce dimensionality
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity_value)
        all_data_2d = tsne.fit_transform(all_data)

        # Separate 2D features based on their original and synthetic labels
        data_ori_0_2d = all_data_2d[:len(data_feat_ori_0)]
        data_ori_1_2d = all_data_2d[len(data_feat_ori_0):len(data_feat_ori_0) + len(data_feat_ori_1)]
        data_syn_0_2d = all_data_2d[
                        len(data_feat_ori_0) + len(data_feat_ori_1):len(data_feat_ori_0) + len(data_feat_ori_1) + len(
                            data_feat_syn_0)]
        data_syn_1_2d = all_data_2d[len(data_feat_ori_0) + len(data_feat_ori_1) + len(data_feat_syn_0):]

        # Plot t-SNE results
        plt.figure(figsize=(6, 4))
        plt.scatter(data_ori_0_2d[:, 0], data_ori_0_2d[:, 1], c='blue', marker='o', alpha=0.1, label='Original Class 0')
        plt.scatter(data_syn_0_2d[:, 0], data_syn_0_2d[:, 1], c='blue', marker='*', label='Synthetic Class 0')
        plt.scatter(data_ori_1_2d[:, 0], data_ori_1_2d[:, 1], c='red', marker='o', alpha=0.1, label='Original Class 1')
        plt.scatter(data_syn_1_2d[:, 0], data_syn_1_2d[:, 1], c='red', marker='*', label='Synthetic Class 1')

        plt.legend()
        plt.title('t-SNE Visualization of Original and Synthetic Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        # Save and display the figure
        plt.savefig(f'tsne_visualization_{self.args.method}_{self.args.dataset}_{self.args.reduction_rate}.pdf',
                    format='pdf')
        print(
            f"Saved figure to tsne_visualization_{self.args.method}_{self.args.dataset}_{self.args.reduction_rate}.pdf")
        plt.show()

    def visualize(args, data):
        """
        Visualizes synthetic and original data using t-SNE and saves the plot as a PDF file.

        Parameters
        ----------
        args : argparse.Namespace
            Command-line arguments and configuration parameters.
        data : Dataset
            The dataset containing the graph data.
        """
        save_path = f'{args.save_path}/reduced_graph/{args.method}'
        feat_syn = torch.load(
            f'{save_path}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location='cpu')
        labels_syn = torch.load(
            f'{save_path}/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location='cpu')
        try:
            adj_syn = torch.load(
                f'{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location=args.device)
        except:
            adj_syn = torch.eye(feat_syn.size(0), device=args.device)

        # Obtain embeddings
        data.adj_train = to_tensor(data.adj_train)
        data.pre_conv = normalize_adj_tensor(data.adj_train, sparse=True)
        data.pre_conv = matmul(data.pre_conv, data.pre_conv)
        feat_train_agg = matmul(data.pre_conv, data.feat_train).float()

        adj_syn = to_tensor(data.adj_syn)
        pre_conv_syn = normalize_adj_tensor(adj_syn, sparse=True)
        pre_conv_syn = matmul(pre_conv_syn, pre_conv_syn)
        feat_syn_agg = matmul(pre_conv_syn, labels_syn).float()

        self.tsne_vis(data.feat_train, data.labels_train, feat_syn, labels_syn)  # Visualizes feature
        self.tsne_vis(feat_train_agg, data.labels_train, feat_syn_agg, labels_syn)  # Visualizes embedding
