from arguments.test_args import get_args
import torch
from models import net_resolution

if __name__ == '__main__':
    args = get_args()
    weights = torch.load(args.srnet_pth)
    model = net_resolution.get_model()
    model.load_state_dict(weights['net'])
    print(weights)
