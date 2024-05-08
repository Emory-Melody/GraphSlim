import csv
import os
import pickle as pkl
from itertools import product
from pathlib import Path

from scipy.stats import pearsonr
from tqdm import tqdm

from graphslim.evaluation.eval_agent import Evaluator


def save_csv(file_path, num):
    with file_path.open(mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(num)
    print("saved csv", file_path)


def load_csv(file_path):
    with file_path.open(mode='r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data


def save_pkl(file_path, data):
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data


class NasEvaluator:
    def __init__(self, args):
        self.args = args
        self.best_params_syn, self.best_params_ori = None, None
        self.results_syn, self.results_ori = [], []

        # Construct parameter combinations as different architecture
        ks = [2, 4, 6, 8, 10]
        nhids = [16, 32, 64, 128, 256, 512]
        alphas = [0.1, 0.2]
        activations = ['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leakyrelu', 'relu6',
                       'elu']
        # ks = [2]
        # nhids = [16]
        # alphas = [0.1]
        # activations = ['sigmoid', 'tanh']

        self.parameter_combinations = list(product(ks, nhids, alphas, activations))

    def evaluate_ori(self, data):
        # train and eval model architecture
        best_acc_val_syn, best_acc_val_ori = 0, 0
        for params in tqdm(self.parameter_combinations):
            args = self.args
            args.run_evaluation = 1
            args.nlayers, args.hidden, args.alpha, args.activation = params

            evaluator = Evaluator(args)
            acc_val_ori, _ = evaluator.nas_evaluate(data, model_type='APPNP', reduced=False, verbose=False)
            self.results_ori.append(acc_val_ori)
            # record best architecture (params) based on val results
            if acc_val_ori > best_acc_val_ori:
                best_acc_val_ori = acc_val_ori
                self.best_params_ori = params

        # save the eval result
        save_path = Path(f'./output')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = Path(f'./output/{self.args.dataset}_results_ori.csv')
        save_csv(file_path, self.results_ori)
        file_path = Path(f'./output/{self.args.dataset}_best_params_ori.pkl')
        save_pkl(file_path, self.best_params_ori)

        # self.cal_pearson()
        # self.test_params_ori(data)

    def evaluate_syn(self, data):
        # train and eval model architecture
        best_acc_val_syn, best_acc_val_ori = 0, 0
        for params in tqdm(self.parameter_combinations):
            args = self.args
            args.run_evaluation = 1
            args.nlayers, args.hidden, args.alpha, args.activation = params

            evaluator = Evaluator(args)
            acc_val_syn, _ = evaluator.nas_evaluate(data, model_type='APPNP', reduced=True, verbose=False)
            self.results_syn.append(acc_val_syn)
            # record best architecture (params) based on val results
            if acc_val_syn > best_acc_val_syn:
                best_acc_val_syn = acc_val_syn
                self.best_params_syn = params

        file_path = Path(f'./output/{self.args.dataset}_{self.args.method}_results_syn.csv')
        save_csv(file_path, self.results_syn)
        file_path = Path(f'./output/{self.args.dataset}_{self.args.method}_best_params_syn.pkl')
        save_pkl(file_path, self.best_params_syn)

    def test_params_ori(self, data):
        if self.best_params_ori is None:
            file_path = Path(f'./output/{self.args.dataset}_best_params_ori.pkl')
            self.best_params_ori = load_pkl(file_path)
        print("best_params_ori", self.best_params_ori)
        args = self.args
        args.run_evaluation = 10
        args.nlayers, args.hidden, args.alpha, args.activation = self.best_params_ori
        evaluator = Evaluator(args)
        acc_test_ori, _ = evaluator.evaluate(data, model_type='APPNP', reduced=False, verbose=False)
        print("NAS: test accuracy on ori graph:", acc_test_ori)

    def test_params_syn(self, data):
        if self.best_params_syn is None:
            file_path = Path(f'./output/{self.args.dataset}_{self.args.method}_best_params_syn.pkl')
            self.best_params_syn = load_pkl(file_path)
        print("best_params_syn", self.best_params_syn)
        args = self.args
        args.run_evaluation = 10
        args.nlayers, args.hidden, args.alpha, args.activation = self.best_params_syn
        evaluator = Evaluator(args)
        acc_test_syn, _ = evaluator.evaluate(data, model_type='APPNP', reduced=False, verbose=False)
        print("NAS: test accuracy on syn graph:", acc_test_syn)

    def get_rank(self, results):
        sorted_tuples = sorted(enumerate(results), key=lambda x: x[1], reverse=True)
        rank_count = 1

        # 生成排名字典，键为元素值，值为排名
        rank_dict = {}
        for _, value in sorted_tuples:
            if value not in rank_dict:
                rank_dict[value] = rank_count
            rank_count += 1

        # 使用排名字典获取每个元素的排名
        ranks = [rank_dict[value] for value in results]

        return ranks

    def cal_pearson(self):
        if len(self.results_syn) == 0 or len(self.results_ori) == 0:
            self.results_syn = [float(x) for x in load_csv(
                Path(f'./output/{self.args.dataset}_{self.args.method}_results_syn.csv'))[0]]
            self.results_ori = [float(x) for x in load_csv(Path(f'./output/{self.args.dataset}_results_ori.csv'))[0]]
        print("ori acc:", self.results_ori)
        print("syn acc:", self.results_syn)
        pearson_corr, p_value = pearsonr(self.results_syn, self.results_ori)
        print("pearson score of accuracy:", pearson_corr)

        results_syn_ranked = self.get_rank(self.results_syn)
        results_ori_ranked = self.get_rank(self.results_ori)
        print("ori rank", results_ori_ranked)
        print("syn rank", results_syn_ranked)
        pearson_corr, p_value = pearsonr(results_syn_ranked, results_ori_ranked)
        print("pearson score of rank:", pearson_corr)
