import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
if os.path.abspath('../..') not in sys.path:
    sys.path.append(os.path.abspath('../..'))

from graphslim.configs import cli
from graphslim.dataset import *
from graphslim.evaluation import *
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    data = get_dataset(args.dataset, args)

    save_path = f'{args.save_path}/reduced_graph/{args.method}'
    feat_syn = torch.load(
        f'{save_path}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location='cpu')
    labels_syn = torch.load(
        f'{save_path}/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location='cpu')

    print(data.feat_train.shape)
    print(data.labels_train.shape)
    print(feat_syn.shape)
    print(labels_syn.shape)

    labels_train_np = data.labels_train.cpu().numpy()
    feat_train_np = data.feat_train.cpu().numpy()
    labels_syn_np = labels_syn.cpu().numpy()
    feat_syn_np = feat_syn.cpu().numpy()

    data_feat_ori_0 = feat_train_np[labels_train_np == 0]
    data_feat_ori_1 = feat_train_np[labels_train_np == 1]

    data_feat_syn_0 = feat_syn_np[labels_syn_np == 0]
    data_feat_syn_1 = feat_syn_np[labels_syn_np == 1]

    all_data = np.concatenate((data_feat_ori_0, data_feat_ori_1, data_feat_syn_0, data_feat_syn_1), axis=0)
    perplexity_value = min(30, len(all_data) - 1)

    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity_value)
    all_data_2d = tsne.fit_transform(all_data)

    data_ori_0_2d = all_data_2d[:len(data_feat_ori_0)]
    data_ori_1_2d = all_data_2d[len(data_feat_ori_0):len(data_feat_ori_0) + len(data_feat_ori_1)]
    data_syn_0_2d = all_data_2d[
                    len(data_feat_ori_0) + len(data_feat_ori_1):len(data_feat_ori_0) + len(data_feat_ori_1) + len(
                        data_feat_syn_0)]
    data_syn_1_2d = all_data_2d[len(data_feat_ori_0) + len(data_feat_ori_1) + len(data_feat_syn_0):]

    # plot
    plt.figure(figsize=(6, 4))

    plt.scatter(data_ori_0_2d[:, 0], data_ori_0_2d[:, 1], c='blue', marker='o', alpha=0.1, label='Original Class 0')
    plt.scatter(data_syn_0_2d[:, 0], data_syn_0_2d[:, 1], c='blue', marker='*', label='Synthetic Class 0')
    plt.scatter(data_ori_1_2d[:, 0], data_ori_1_2d[:, 1], c='red', marker='o', alpha=0.1, label='Original Class 1')
    plt.scatter(data_syn_1_2d[:, 0], data_syn_1_2d[:, 1], c='red', marker='*', label='Synthetic Class 1')

    plt.legend()
    plt.title('t-SNE visualization of original and synthetic data')
    plt.xlabel('feature of class 1')
    plt.ylabel('feature of class 2')

    # save figure to pdf
    plt.savefig(f'tsne_visualization_{args.method}_{args.dataset}_{args.reduction_rate}.pdf', format='pdf')
    print(f"save fig to tsne_visualization_{args.method}_{args.dataset}_{args.reduction_rate}.pdf")

    plt.show()
