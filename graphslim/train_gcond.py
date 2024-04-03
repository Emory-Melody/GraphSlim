from condensation import *
from configs import *
from dataset import *

args = cli(standalone_mode=False)
data = get_dataset(args.dataset, normalize_features=args.normalize_features, transform=None)
agent = router_condense(data, args)
agent.train()

# agent.cross_architecture_eval()
