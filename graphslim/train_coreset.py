from configs import cli
from configs import load_config
from graphslim.dataset import *
from graphslim.sparsification import router_select

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    # load specific augments
    args = load_config(args)

    data = get_dataset(args.dataset, args)

    result = router_select(data, args)
