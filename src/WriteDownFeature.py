import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models import fusion_model1, sface, edsr

import os
import numpy as np
from tqdm import tqdm
from arguments import train_args
from util import common
from loaders import celeba_loader

def initModels():
    ## Setup FNet
    fnet = sface.sface()
    fnet.load_state_dict(torch.load('../../pretrained/sface.pth'))
    fnet.to(args.device)
    common.freeze(fnet)
    srnet = edsr.Edsr()
    srnet.load_state_dict(torch.load('/content/drive/MyDrive/app/test_raw/backup_epoch15.pth')['net'])
    srnet.to(args.device)
    common.freeze(srnet)
    lr_fnet = sface.SphereFace()
    lr_fnet.load_state_dict(torch.load('/content/drive/MyDrive/app/3_13_learn_guide_train/learn_guide_backup_epoch18.pth')['net'])
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
    dataloader = celeba_loader.get_loader_downsample(args)

    bar = tqdm(dataloader, total=len(dataloader), ncols=0)
    all_data = None
    count = 0
    for batch_id, inputs in enumerate(bar):
        target = inputs['id'].to(args.device)
        data = torch.reshape(target, [-1, 1])
        hr_face = inputs['down1'].to(args.device)
        data = torch.cat([data, fnet(hr_face)], dim=1)
        for i in range(1, 4):
            lr_face = inputs['down{}'.format(2 ** i)].to(args.device)
            feature1, feature2 = fusion_model1.getFeatures(srnet, fnet, lr_fnet, lr_face)
            data = torch.cat([data, feature1, feature2], dim=1)
        count += 1
        if all_data is None:
            all_data = data.cpu()
        else:
            all_data = torch.cat([all_data, data.cpu()], dim=0)
    torch.save(all_data, "../../celeba_features.pth")