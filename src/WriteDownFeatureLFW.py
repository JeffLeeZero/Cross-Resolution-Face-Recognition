import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models import fusion_model, sface, edsr_se

import os
import numpy as np
from tqdm import tqdm
from arguments import train_args
from util import common
from loaders import celeba_loader, lfw_loader


def initModels():
    ## Setup FNet
    fnet = sface.sface()
    fnet.load_state_dict(torch.load('../../pretrained/sface.pth'))
    fnet.to(args.device)
    common.freeze(fnet)
    srnet = edsr_se.Edsr()
    srnet.load_state_dict(torch.load('/content/drive/MyDrive/app/edsrse_100_4_13/backup.pth')['net'])
    srnet.to(args.device)
    common.freeze(srnet)
    lr_fnet = sface.SeSface()
    lr_fnet.load_state_dict(torch.load('/content/drive/MyDrive/app2/sesface_4_13/sesface_backup_epoch13.pth')['net'])
    lr_fnet.to(args.device)
    lr_fnet.setVal(True)
    common.freeze(lr_fnet)
    fnet.eval()
    srnet.eval()
    lr_fnet.eval()
    return fnet, srnet, lr_fnet


if __name__ == '__main__':
    args = train_args.get_args()
    fnet, srnet, lr_fnet = initModels()
    dataloader = lfw_loader.get_loader(-1, 8, 96, 112, args.bs)

    bar = tqdm(dataloader, total=len(dataloader), ncols=0)
    all_data = None
    for batch_id, inputs in enumerate(bar):
        img1, _, img1_flip, _, _ = inputs
        img1 = img1.to(args.device)
        img1_flip = img1_flip.to(args.device)
        down_f = torch.ones(size=(args.bs, 2, 1, 1)).to('cuda:0')
        feature1, feature2 = fusion_model.getFeatures(srnet, fnet, lr_fnet, img1, down_f)
        data = torch.cat([feature1, feature2], dim=1)
        feature1, feature2 = fusion_model.getFeatures(srnet, fnet, lr_fnet, img1_flip, down_f)
        data = torch.cat([data, feature1, feature2], dim=1)
        if all_data is None:
            all_data = data.cpu()
        else:
            all_data = torch.cat([all_data, data.cpu()], dim=0)
    torch.save(all_data, "../../lfw_features.pth")