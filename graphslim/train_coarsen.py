from coarsening import *
from configs import cli
from dataset import *

if __name__ == '__main__':
    path = "checkpoints/"
    if not os.path.isdir(path):
        os.mkdir(path)
    # TODO: do we need a dictionary to transfer the different reduction ratios?
    args = cli(standalone_mode=False)
    data = get_dataset(args.dataset, return_pyg=True)
    # TODO: change to router_coarsening
    agent = CoarseningBase(data, args, device='cuda')

    agent.train()
