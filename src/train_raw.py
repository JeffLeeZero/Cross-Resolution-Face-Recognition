import torch

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from arguments import train_args
from loaders import celeba_loader
from models import sface, edsr
from util import common
import numpy as np
import lfw_verification as val
import os

from torchsummary import summary


def save_network(args, net, which_step):
    save_filename = 'edsr_lambda{}_step{}.pth'.format(args.lamb_id, which_step)
    save_dir = os.path.join(args.checkpoints_dir, args.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_filename)
    if len(args.gpu_ids) > 1 and torch.cuda.is_available():
        try:
            torch.save(net.module.cpu().state_dict(), save_path)
        except:
            torch.save(net.cpu().state_dict(), save_path)
    else:
        torch.save(net.cpu().state_dict(), save_path)


def common_init(args):
    ## Setup SRNet
    srnet = edsr.Edsr()
    srnet.to(args.device)
    if len(args.gpu_ids) > 1:
        srnet = nn.DataParallel(srnet)

    optimizer = optim.Adam(srnet.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=0.5)
    last_epoch = -1
    return srnet, optimizer, last_epoch, scheduler


def backup_init(args):
    save_filename = 'backup.pth'
    save_dir = os.path.join(args.backup_dir, args.name)
    save_path = os.path.join(save_dir, save_filename)
    checkpoint = torch.load(save_path)  # 加载断点

    ## Setup SRNet
    srnet = edsr.Edsr()
    srnet.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    srnet.to(args.device)
    if len(args.gpu_ids) > 1:
        srnet = nn.DataParallel(srnet)

    optimizer = optim.Adam(srnet.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数

    last_epoch = checkpoint['epoch']  # 设置开始的epoch

    scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=0.5, last_epoch=last_epoch)
    scheduler.load_state_dict(checkpoint['scheduler'])
    return srnet, optimizer, last_epoch, scheduler


def save_network_for_backup(args, srnet, optimizer, scheduler, epoch_id):
    checkpoint = {
        "net": srnet.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch_id,
        'scheduler': scheduler.state_dict()
    }

    save_filename = 'backup.pth'
    save_dir = os.path.join(args.backup_dir, args.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(checkpoint, save_path)


def train():
    dataloader = celeba_loader.get_loader_downsample(args)
    train_iter = iter(dataloader)
    ## Setup FNet
    fnet = sface.sface()

    fnet.load_state_dict(torch.load('../../pretrained/sface.pth'))
    common.freeze(fnet)
    fnet.to(args.device)

    if args.Continue:
        srnet, optimizer, last_epoch, scheduler = backup_init(args)
    else:
        srnet, optimizer, last_epoch, scheduler = common_init(args)
    summary(fnet, (3,96,112))
    criterion_pixel = nn.L1Loss()

    losses = ['loss', 'loss_pixel', 'loss_feature', 'lr', 'index']
    epochs = args.epoch
    best_acc = 0.927
    for epoch_id in range(last_epoch + 1, epochs):
        bar = tqdm(dataloader, total=len(dataloader), ncols=0)
        loss = [0.0, 0.0, 0.0, 0.0, 0.0]
        loss_pixel = [0.0, 0.0, 0.0, 0.0, 0.0]
        loss_feature = [0.0, 0.0, 0.0, 0.0, 0.0]
        count = [0, 0, 0, 0, 0]
        srnet.train()
        for _, inputs in enumerate(bar):
            lr = optimizer.param_groups[0]['lr']
            index = np.random.randint(2, 4 + 1)
            lr_face = inputs['down{}'.format(2 ** index)].to(args.device)
            mr_face = inputs['down{}'.format(2 ** (index - 2))].to(args.device)
            if index == 2:
                hr_face = mr_face
            else:
                hr_face = inputs['down1'].to(args.device)
            sr_face = srnet(lr_face)
            lossd_pixel = criterion_pixel(sr_face, mr_face.detach())
            loss_pixel[index] += lossd_pixel.item()
            lossd = lossd_pixel
            # Feature loss
            sr_face_up = nn.functional.interpolate(sr_face, size=(112, 96), mode='bilinear', align_corners=False)
            if args.lamb_id > 0:
                sr_face_feature = fnet(common.tensor2SFTensor(sr_face_up))
                hr_face_feature = fnet(common.tensor2SFTensor(hr_face)).detach()
                lossd_feature = 1 - torch.nn.CosineSimilarity()(sr_face_feature, hr_face_feature)
                lossd_feature = lossd_feature.mean()
                loss_feature[index] += lossd_feature.item()

            lossd += args.lamb_id * lossd_feature
            loss[index] += lossd.item()
            count[index] += 1
            optimizer.zero_grad()
            lossd.backward()
            optimizer.step()
            scheduler.step()  # update learning rate
            # display
            description = "epoch {} :".format(epoch_id)
            description += 'loss: {:.4f} '.format(loss[index] / count[index])
            description += 'loss_pixel: {:.4f} '.format(loss_pixel[index] / count[index])
            description += 'loss_feature: {:.4f} '.format(loss_feature[index] / count[index])
            description += 'lr: {:.3e} '.format(lr)
            description += 'index: {:.0f} '.format(index)
            bar.set_description(desc=description)

        acc = val.val_raw("sface", -1, 8, 96, 112, 32, args.device, fnet, srnet)
        if acc >= best_acc or epoch_id % 3 == 0:
            best_acc = acc
            save_network_for_backup(args, srnet, optimizer, scheduler, epoch_id)

    # Save the final SR model
    save_network(args, srnet, args.iterations)


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
        train()
    else:
        eval()
