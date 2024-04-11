from configs import cli
from dataset import *
from graphslim.coarsening import *

if __name__ == '__main__':
    # TODO: do we need a dictionary to transfer the different reduction ratios?
    args = cli(standalone_mode=False)
    data = get_dataset(args.dataset, args)
    # TODO: change to router_coarsening
    agent = router_coarse(data, args)

    agent.train()
