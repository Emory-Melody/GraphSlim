from condensation import *
from configs import *
from dataset import *

args = cli(standalone_mode=False)
data = get_dataset(args.dataset, args)
agent = router_condense(data, args)
agent.train()

# agent.cross_architecture_eval()
