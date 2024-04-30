import csv
from itertools import product
from pathlib import Path

from scipy.stats import pearsonr
from tqdm import tqdm

from graphslim.evaluation.eval_agent import Evaluator


def csv_writer(file_path, num):
    with file_path.open(mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(num)


def csv_reader(file_path):
    with file_path.open(mode='r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data


class NasEvaluator:
    def __init__(self, args):
        self.args = args
        self.best_params_syn, self.best_params_ori = None, None
        self.results_syn, self.results_ori = [], []

        # Construct parameter combinations as different architecture
        ks = [2, 4, 6, 8, 10]  # [2, 4, 6, 8, 10]
        nhids = [16, 32, 64, 128, 256, 512]  # [16, 32, 64, 128, 256, 512]
        alphas = [0.1, 0.2]  # [0.1, 0.2]
        activations = ['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leakyrelu', 'relu6',
                       'elu']  # ['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leakyrelu', 'relu6', 'elu']  #
        self.parameter_combinations = list(product(ks, nhids, alphas, activations))

    def evaluate_ori(self, data):
        # train and eval model architecture
        best_acc_val_syn, best_acc_val_ori = 0, 0
        for params in tqdm(self.parameter_combinations):
            args = self.args
            args.runs = 1
            args.nlayers, args.eval_hidden, args.alpha, args.activation = params

            evaluator = Evaluator(args)
            acc_val_ori, _ = evaluator.nas_evaluate(data, model_type='APPNPRich', reduced=False, verbose=False)
            self.results_ori.append(acc_val_ori)
            # record best architecture (params) based on val results
            if acc_val_ori > best_acc_val_ori:
                best_acc_val_ori = acc_val_ori
                self.best_params_ori = params

        # save the eval result
        file_path = Path(f'{self.args.dataset}_results_ori.csv')
        csv_writer(file_path, self.results_ori)

        # self.cal_pearson()
        # self.test_params_ori(data)

    def evaluate_syn(self, data):
        # train and eval model architecture
        best_acc_val_syn, best_acc_val_ori = 0, 0
        for params in tqdm(self.parameter_combinations):
            args = self.args
            args.runs = 1
            args.nlayers, args.eval_hidden, args.alpha, args.activation = params

            evaluator = Evaluator(args)
            acc_val_syn, _ = evaluator.nas_evaluate(data, model_type='APPNPRich', reduced=True, verbose=False)
            self.results_syn.append(acc_val_syn)
            # record best architecture (params) based on val results
            if acc_val_syn > best_acc_val_syn:
                best_acc_val_syn = acc_val_syn
                self.best_params_syn = params

        # save the eval result
        file_path = Path(f'{self.args.dataset}_{self.args.method}_{self.args.reduction_rate}_results_syn.csv')
        csv_writer(file_path, self.results_syn)

    def test_params_ori(self, data):
        print("best_params_ori", self.best_params_ori)
        args = self.args
        args.runs = 1
        args.nlayers, args.eval_hidden, args.alpha, args.activation = self.best_params_ori
        evaluator = Evaluator(args)
        acc_test_ori, _ = evaluator.evaluate(data, model_type='APPNPRich', reduced=False, verbose=False)
        print("NAS: test accuracy on ori graph:", acc_test_ori)

    def test_params_syn(self, data):
        print("best_params_syn", self.best_params_syn)
        args = self.args
        args.runs = 1
        args.nlayers, args.eval_hidden, args.alpha, args.activation = self.best_params_syn
        evaluator = Evaluator(args)
        acc_test_syn, _ = evaluator.evaluate(data, model_type='APPNPRich', reduced=True, verbose=False)
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
        if len(self.results_syn) == 0:
            self.results_syn = [float(x) for x in csv_reader(
                Path(f'{self.args.dataset}_{self.args.method}_{self.args.reduction_rate}_results_syn.csv'))[0]]
            self.results_ori = [float(x) for x in csv_reader(Path(f'{self.args.dataset}_results_ori.csv'))[0]]
        print("ori acc:", self.results_ori)
        print("syn acc:", self.results_syn)
        pearson_corr, p_value = pearsonr(self.results_syn, self.results_ori)
        print("pearson score of accuracy:", pearson_corr)

        # results_syn_ranked = rankdata(self.results_syn, method='max')
        # results_ori_ranked = rankdata(self.results_ori, method='max')
        results_syn_ranked = self.get_rank(self.results_syn)
        results_ori_ranked = self.get_rank(self.results_ori)
        print("ori rank", results_ori_ranked)
        print("syn rank", results_syn_ranked)
        pearson_corr, p_value = pearsonr(results_syn_ranked, results_ori_ranked)
        print("pearson score of rank:", pearson_corr)


    def cal_hit_index(self, results_syn=None, results_ori=None):
        # Read the CSV files and get the first row as a list of numbers
        if results_syn is None and results_ori is None:
            results_syn = [float(x) for x in csv_reader(Path('results_syn.csv'))[0]]
            results_ori = [float(x) for x in csv_reader(Path('results_ori.csv'))[0]]

        # Find the index of the top value in results_ori
        top_index_ori = results_ori.index(max(results_ori))

        # Sort results_syn while retaining original indices
        sorted_results_syn_indices = sorted(range(len(results_syn)), key=lambda k: results_syn[k])

        # Find where the top index from results_ori hits in the sorted results_syn indices
        hit_position = sorted_results_syn_indices.index(top_index_ori)

        # Output the original lists and the hit position
        print("Original results_syn:", results_syn)
        print("Original results_ori:", results_ori)
        print("Hit position:", hit_position + 1)  # Adding 1 to make it 1-based index

    # def best_nas_results(self, results_syn=None, results_ori=None):
    #     if results_syn is None and results_ori is None:
    #         results_syn = [float(x) for x in csv_reader(Path('results_syn.csv'))[0]]
    #         results_ori = [float(x) for x in csv_reader(Path('results_ori.csv'))[0]]
    #
    #     top_index_syn = results_syn.index(max(results_syn))
    #     acc_syn = results_syn[top_index_syn]
    #     acc_ori = max(results_ori)
    #
    #     print("Best :", acc_syn)
