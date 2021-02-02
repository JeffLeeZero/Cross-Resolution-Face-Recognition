import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models import net_resolution, sface, edsr

import os
import numpy as np
from tqdm import tqdm
from arguments import train_args
from util import common
from loaders import celeba_loader


def save_network(args, net, epoch):
    save_filename = 'net_resolution_epoch{}.pth'.format(epoch)
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


def tensor2SFTensor(tensor):
    return tensor * (127.5 / 128.)


def common_init(args):
    net = net_resolution.get_pretrain_modle(srnet_path="../../pretrained/srnet.pth", fnet_path="../../pretrained/sface.pth")
    net.freeze("convs")
    net.freeze("srnet")
    net.to(args.device)
    if len(args.gpu_ids) > 1:
        net = nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=0.5)
    last_epoch = -1
    return net, optimizer, last_epoch, scheduler


def backup_init(args):
    save_filename = 'backup.pth'
    save_dir = os.path.join(args.backup_dir, args.name)
    save_path = os.path.join(save_dir, save_filename)
    checkpoint = torch.load(save_path)  # 加载断点

    ## Setup SRNet
    net = net_resolution.get_model()
    net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    net.freeze("convs")
    net.freeze("srnet")
    net.to(args.device)
    if len(args.gpu_ids) > 1:
        srnet = nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数

    last_epoch = checkpoint['epoch']  # 设置开始的epoch

    scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=0.5, last_epoch=last_epoch)
    return net, optimizer, last_epoch, scheduler


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


def main():
    args = train_args.get_args()
    dataloader = celeba_loader.get_loader_downsample(args)
    ## Setup FNet
    fnet = sface.sface()
    fnet.load_state_dict(torch.load('../../pretrained/sface.pth'))
    fnet.to(args.device)
    if args.Continue:
        net, optimizer, last_epoch, scheduler = backup_init(args)
    else:
        net, optimizer, last_epoch, scheduler = common_init(args)

    criterion_pixel = nn.L1Loss()

    losses = ['loss',  'lr', 'index']
    epochs = args.epoch
    for epoch_id in range(last_epoch + 1, epochs):
        bar = tqdm(dataloader, total=len(dataloader), ncols=0)
        loss = [0.0, 0.0, 0.0]
        for batch_id, inputs in enumerate(bar):
            net.train()
            scheduler.step()  # update learning rate
            lr = optimizer.param_groups[0]['lr']
            index = np.random.randint(2, 4 + 1)
            lr_face = inputs['down{}'.format(2 ** index)].to(args.device)
            mr_face = inputs['down{}'.format(2 ** (index - 2))].to(args.device)
            if index == 2:
                hr_face = mr_face
            else:
                hr_face = inputs['down1'].to(args.device)

            w = torch.full([lr_face.shape[0], 1], lr_face.shape[2]).to(args.device)
            h = torch.full([lr_face.shape[0], 1], lr_face.shape[3]).to(args.device)
            sr_face_feature, _ = net(lr_face, w, h)
            hr_face_feature = fnet(tensor2SFTensor(hr_face)).detach()
            loss_feature = 1 - torch.nn.CosineSimilarity()(sr_face_feature, hr_face_feature)
            loss_feature = loss_feature.mean()
            loss[index-2] += loss_feature.float()
            optimizer.zero_grad()
            loss_feature.backward()
            optimizer.step()
            # display
            description = ""
            for name in losses:
                try:
                    value = float(eval(name))
                    if name == 'index':
                        description += '{}: {:.0f} '.format(name, value)
                    elif name == 'lr':
                        description += '{}: {:.3e} '.format(name, value)
                    else:
                        description += '{}: {:.3f} '.format(name, value / (batch_id + 1))
                except:
                    continue
            bar.set_description(desc=description)

        save_network_for_backup(args, net, optimizer, scheduler, epoch_id)

    # Save the final SR model
    save_network(args, net, epochs)


if __name__ == '__main__':
    main()
