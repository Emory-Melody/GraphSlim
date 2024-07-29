import csv
import os
import pickle as pkl
from itertools import product
from pathlib import Path

from scipy.stats import pearsonr
from tqdm import tqdm

from graphslim.evaluation.eval_agent import Evaluator


def save_csv(file_path, num):
    file_path = Path(file_path)
    with file_path.open(mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(num)
    print("saved csv", file_path)


def load_csv(file_path):
    file_path = Path(file_path)
    with file_path.open(mode='r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data


def save_pkl(file_path, data):
    file_path = Path(file_path)
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)


def load_pkl(file_path):
    file_path = Path(file_path)
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data


class NasEvaluator:
    """
    Class for evaluating neural architecture search (NAS) performance on original and synthetic graphs.
    """
    def __init__(self, args):
        self.args = args
        self.best_params_syn, self.best_params_ori = None, None
        self.results_syn, self.results_ori = [], []

        # Define possible values for parameters to search over
        ks = [2, 4, 6, 8, 10]
        nhids = [16, 32, 64, 128, 256, 512]
        alphas = [0.1, 0.2]
        activations = ['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leakyrelu', 'relu6', 'elu']
        # ks = [2, 4, 6]
        # nhids = [16, 32]
        # alphas = [0.1]
        # activations = ['relu']

        self.save_path = f'checkpoints/nas/{args.dataset}'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.parameter_combinations = list(product(ks, nhids, alphas, activations))

    def evaluate_ori(self, data):
        """
        Evaluates various architectures on the original graph and identifies the best one.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        """
        best_acc_val_ori = 0
        for params in tqdm(self.parameter_combinations):
            args = self.args
            args.run_evaluation = 1
            args.nlayers, args.hidden, args.alpha, args.activation = params
            args.eval_epochs = 600
            args.ntrans = 2

            evaluator = Evaluator(args)
            acc_val_ori, _ = evaluator.nas_evaluate(data, model_type='APPNP', reduced=False, verbose=False)
            self.results_ori.append(acc_val_ori)

            # Update best architecture based on validation accuracy
            if acc_val_ori > best_acc_val_ori:
                best_acc_val_ori = acc_val_ori
                self.best_params_ori = params

        # Save results to files
        file_path = f'{self.save_path}/results_ori.csv'
        save_csv(file_path, self.results_ori)
        file_path = f'{self.save_path}/best_params_ori.pkl'
        save_pkl(file_path, self.best_params_ori)

    def evaluate_syn(self, data):
        """
        Evaluates various architectures on the synthetic graph and identifies the best one.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        """
        best_acc_val_syn = 0
        for params in tqdm(self.parameter_combinations):
            args = self.args
            args.run_evaluation = 1
            args.nlayers, args.hidden, args.alpha, args.activation = params
            args.ntrans = 2

            evaluator = Evaluator(args)
            acc_val_syn, _ = evaluator.nas_evaluate(data, model_type='APPNP', reduced=True, verbose=False)
            self.results_syn.append(acc_val_syn)

            # Update best architecture based on validation accuracy
            if acc_val_syn > best_acc_val_syn:
                best_acc_val_syn = acc_val_syn
                self.best_params_syn = params

        # Save results to files
        file_path = f'{self.save_path}/{self.args.method}_results_syn.csv'
        save_csv(file_path, self.results_syn)
        file_path = f'{self.save_path}/{self.args.method}_best_params_syn.pkl'
        save_pkl(file_path, self.best_params_syn)

    def test_params_ori(self, data):
        """
        Tests the best architecture on the original graph using the best parameters.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        """
        if self.best_params_ori is None:
            file_path = f'{self.save_path}/best_params_ori.pkl'
            self.best_params_ori = load_pkl(file_path)

        self.args.logger.info(f"Best parameters for original graph: {self.best_params_ori}")
        args = self.args
        args.nlayers, args.hidden, args.alpha, args.activation = self.best_params_ori
        args.eval_epochs = 600
        args.ntrans = 2

        evaluator = Evaluator(args)
        acc_test_ori, _ = evaluator.evaluate(data, model_type='APPNP', reduced=False, verbose=False)
        self.args.logger.info(f"Test accuracy on original graph: {acc_test_ori}")

    def test_params_syn(self, data):
        """
        Tests the best architecture on the synthetic graph using the best parameters.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.

        Returns
        -------
        acc_test_syn : float
            The test accuracy on the synthetic graph.
        """
        if self.best_params_syn is None:
            file_path = f'{self.save_path}/{self.args.method}_best_params_syn.pkl'
            self.best_params_syn = load_pkl(file_path)

        self.args.logger.info(f"Best parameters for synthetic graph: {self.best_params_syn}")
        args = self.args
        args.nlayers, args.hidden, args.alpha, args.activation = self.best_params_syn
        args.eval_epochs = 600
        args.ntrans = 2

        evaluator = Evaluator(args)
        acc_test_syn, _ = evaluator.evaluate(data, model_type='APPNP', reduced=True, verbose=False)
        self.args.logger.info(f"Test accuracy on synthetic graph: {acc_test_syn}")

        return acc_test_syn

    def get_rank(self, results):
        """
        Ranks results based on their values.

        Parameters
        ----------
        results : list of float
            The list of results to rank.

        Returns
        -------
        ranks : list of int
            The list of ranks corresponding to the results.
        """
        sorted_tuples = sorted(enumerate(results), key=lambda x: x[1], reverse=True)
        rank_count = 1

        rank_dict = {}
        for _, value in sorted_tuples:
            if value not in rank_dict:
                rank_dict[value] = rank_count
            rank_count += 1

        ranks = [rank_dict[value] for value in results]

        return ranks

    def cal_pearson(self):
        """
        Calculates Pearson correlation coefficients between synthetic and original results.

        Returns
        -------
        pearson_corr_acc : float
            Pearson correlation coefficient of accuracies.
        pearson_corr_rank : float
            Pearson correlation coefficient of ranks.
        """
        if len(self.results_syn) == 0 or len(self.results_ori) == 0:
            self.results_syn = [float(x) for x in load_csv(f'{self.save_path}/{self.args.method}_results_syn.csv')[0]]
            self.results_ori = [float(x) for x in load_csv(f'{self.save_path}/results_ori.csv')[0]]

        pearson_corr_acc, _ = pearsonr(self.results_syn, self.results_ori)
        self.args.logger.info(f"Pearson correlation of accuracy: {pearson_corr_acc}")

        results_syn_ranked = self.get_rank(self.results_syn)
        results_ori_ranked = self.get_rank(self.results_ori)
        pearson_corr_rank, _ = pearsonr(results_syn_ranked, results_ori_ranked)
        self.args.logger.info(f"Pearson correlation of rank: {pearson_corr_rank}")

        return pearson_corr_acc, pearson_corr_rank
