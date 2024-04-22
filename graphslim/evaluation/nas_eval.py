import csv
from pathlib import Path

from scipy.stats import pearsonr

from graphslim.configs import cli
from graphslim.dataset import *
from graphslim.evaluation.eval_agent import Evaluator
import numpy as np


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
    def __init__(self, data, args):
        self.args = args
        self.data = data

    def evaluate_syn(self):
        results = []
        for k in [2]:  # [2, 4, 6, 8, 10]
            for nhid in [256]:  # [16,32,64,128,256,512]
                for alpha in [0.1]:  # [0.1, 0.2]
                    for activation in ['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leakyrelu', 'relu6', 'elu']:
                        args = self.args
                        args.runs = 1
                        args.nlayers = k
                        args.hidden = nhid
                        args.alpha = alpha
                        args.activation = activation
                        agent = Evaluator(data, args, device='cuda')
                        acc_test = agent.train(model_type='APPNP1')
                        results.append(acc_test)

        file_path = Path('results_syn.csv')
        # results = csv_reader(file_path)
        # results.append(acc_test)
        csv_writer(file_path, results)

    def cal_pearson(self):
        results_syn = csv_reader(Path('results_syn.csv'))[0]
        results_whole = csv_reader(Path('results_whole.csv'))[0]
        print(results_syn)
        print(results_whole)

        pearson_corr, _ = pearsonr(np.array(list(map(float, results_syn))), np.array(list(map(float, results_whole))))
        print("Pearson:", pearson_corr)

    def cal_hit_index(self):
        # Read the CSV files and get the first row as a list of numbers
        results_syn = [int(x) for x in csv_reader(Path('results_syn.csv'))[0]]
        results_whole = [int(x) for x in csv_reader(Path('results_whole.csv'))[0]]

        # Find the index of the top value in results_whole
        top_index_whole = results_whole.index(max(results_whole))

        # Sort results_syn while retaining original indices
        sorted_results_syn_indices = sorted(range(len(results_syn)), key=lambda k: results_syn[k])

        # Find where the top index from results_whole hits in the sorted results_syn indices
        hit_position = sorted_results_syn_indices.index(top_index_whole)

        # Output the original lists and the hit position
        print("Original results_syn:", results_syn)
        print("Original results_whole:", results_whole)
        print("Hit position:", hit_position + 1)  # Adding 1 to make it 1-based index


if __name__ == '__main__':
    args = cli(standalone_mode=False)
    data = get_dataset(args.dataset, args.normalize_features)

    if args.dataset in ['cora', 'citeseer']:
        args.epsilon = 0.05
    else:
        args.epsilon = 0.01

    NasEvaluator = NasEvaluator(data, args)

    # NasEvaluator.evaluate_syn()

    NasEvaluator.cal_pearson()




