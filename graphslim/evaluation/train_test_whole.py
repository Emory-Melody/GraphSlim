from graphslim.configs import cli
from graphslim.configs import load_config
from graphslim.dataset import *
from graphslim.models import *
import csv
from pathlib import Path

def csv_writer(file_path, num):
    with file_path.open(mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(num)

def csv_reader(file_path):
    with file_path.open(mode='r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    # load specific augments
    args = load_config(args)

    data = get_dataset(args.dataset, args.normalize_features)
    results = []
    for k in [2]: # [2, 4, 6, 8, 10]
        for nhid in [256]: # [16,32,64,128,256,512]
            for alpha in [0.1]: # [0.1, 0.2]
                for activation in ['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leakyrelu', 'relu6', 'elu']:

                    model = APPNP1(nfeat=data.feat_full.shape[1], nhid=nhid, nclass=data.nclass, device=args.device,
                                weight_decay=args.weight_decay, nlayers=k, alpha=alpha, activation=activation).to(args.device)

                    if args.setting == 'trans':
                        model.fit_with_val(data.feat_full, data.adj_full, data.labels_train, data, train_iters=args.epochs, verbose=False)
                        acc_test = model.test(data)

                    if args.setting == 'ind':
                        model.fit_with_val(data.feat_train, data.adj_train, data.labels_train, data, train_iters=args.epochs, normalize=True,
                                           verbose=False)
                        model.eval()
                        labels_test = torch.LongTensor(data.labels_test).cuda()
                        feat_test, adj_test = data.feat_test, data.adj_test
                        output = model.predict(feat_test, adj_test)
                        loss_test = F.nll_loss(output, labels_test)
                        acc_test = accuracy(output, labels_test)
                        # print("Test set results:",
                        #       "loss= {:.4f}".format(loss_test.item()),
                        #       "accuracy= {:.4f}".format(acc_test.item()))
                    results.append(acc_test)

    file_path = Path('evaluation/results_whole.csv')
    # results = csv_reader(file_path)
    # results.append(acc_test)
    csv_writer(file_path, results)


