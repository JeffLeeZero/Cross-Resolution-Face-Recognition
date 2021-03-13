import torch

from arguments import train_args
from loaders import celeba_loader
from models import sface, edsr
import lfw_verification as val

def eval():
    dataloader = celeba_loader.get_loader_downsample(args)
    ## Setup FNet
    fnet = sface.sface()
    fnet.load_state_dict(torch.load('../../pretrained/sface.pth'))
    fnet.to(args.device)
    srnet = edsr.Edsr()
    srnet.load_state_dict(torch.load(args.model_file))
    srnet.to(args.device)

    val.val_raw("sface", -1, 16, 96, 112, 32, args.device, fnet, srnet)
    val.val_raw("sface", -1, 12, 96, 112, 32, args.device, fnet, srnet)
    val.val_raw("sface", -1, 8, 96, 112, 32, args.device, fnet, srnet)
    val.val_raw("sface", -1, 6, 96, 112, 32, args.device, fnet, srnet)
    val.val_raw("sface", -1, 4, 96, 112, 32, args.device, fnet, srnet)



args = train_args.get_args()
if __name__ == '__main__':
    if args.type == 'train':
        print('train')
    else:
        eval()
