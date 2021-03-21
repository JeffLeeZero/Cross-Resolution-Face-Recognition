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
import lfw_verification as val
from losses.fusion_loss import FusionLoss


def save_network(args, net, epoch):
    save_filename = 'fusion1_epoch{}.pth'.format(epoch)
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
    net = fusion_model1.FusionModel2()
    net.to(args.device)
    if len(args.gpu_ids) > 1:
        net = nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=0.5)
    last_epoch = -1
    return net, optimizer, last_epoch, scheduler


def backup_init(args):
    checkpoint = torch.load(args.model_file)  # 加载断点

    net = fusion_model1.FusionModel2()
    net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    net.to(args.device)
    if len(args.gpu_ids) > 1:
        srnet = nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数

    last_epoch = checkpoint['epoch']  # 设置开始的epoch

    scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=0.5, last_epoch=last_epoch)
    scheduler.load_state_dict(checkpoint['scheduler'])
    return net, optimizer, last_epoch, scheduler


def save_network_for_backup(args, srnet, optimizer, scheduler, epoch_id):
    checkpoint = {
        "net": srnet.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch_id,
        'scheduler': scheduler.state_dict()
    }

    save_filename = 'fusion1_backup_epoch{}.pth'.format(epoch_id)
    save_dir = os.path.join(args.backup_dir, args.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(checkpoint, save_path)


def initModels():
    ## Setup FNet
    fnet = sface.sface()
    fnet.load_state_dict(torch.load('../../pretrained/sface.pth'))
    fnet.to(args.device)
    common.freeze(fnet)
    srnet = edsr.Edsr()
    srnet.load_state_dict(torch.load('/content/drive/MyDrive/app/test_raw/backup_epoch11.pth')['net'])
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


def main():
    dataloader = celeba_loader.get_loader_downsample(args)
    fnet, srnet, lr_fnet = initModels()
    if args.Continue:
        net, optimizer, last_epoch, scheduler = backup_init(args)
    else:
        net, optimizer, last_epoch, scheduler = common_init(args)

    best_acc = 0.0
    epochs = args.epoch
    criterion = FusionLoss()
    for epoch_id in range(last_epoch + 1, epochs):
        bar = tqdm(dataloader, total=len(dataloader), ncols=0)
        loss = [0.0, 0.0, 0.0, 0.0, 0.0]
        loss_class = [0.0, 0.0, 0.0, 0.0, 0.0]
        loss_feature = [0.0, 0.0, 0.0, 0.0, 0.0]
        count = [0, 0, 0, 0, 0]
        net.train()
        for batch_id, inputs in enumerate(bar):
            lr = optimizer.param_groups[0]['lr']
            index = np.random.randint(1, 3 + 1)
            lr_face = inputs['down{}'.format(2 ** index)].to(args.device)
            hr_face = inputs['down1'].to(args.device)
            target = inputs['id'].to(args.device)
            feature1, feature2 = fusion_model1.getFeatures(srnet, fnet, lr_fnet, lr_face)
            target_feature = fnet(common.tensor2SFTensor(hr_face)).detach()
            feature, classes = net(torch.cat([feature1, feature2], dim=1))
            lossd, lossd_class, lossd_feature = criterion(classes, target, feature, target_feature)
            loss[index] += lossd.item()
            loss_class[index] += lossd_class
            loss_feature[index] += lossd_feature
            count[index] += 1
            optimizer.zero_grad()
            lossd.backward()
            optimizer.step()
            scheduler.step()  # update learning rate
            # display
            description = "epoch {} : ".format(epoch_id)
            description += 'loss: {:.4f} '.format(loss[index] / count[index])
            description += 'loss_class: {:.4f} '.format(loss_class[index] / count[index])
            description += 'loss_feature: {:.4f} '.format(loss_feature[index] / count[index])
            description += 'lr: {:.3e} '.format(lr)
            description += 'index: {:.0f} '.format(index)
            bar.set_description(desc=description)
        acc = val.fusion_val(-1, 8, 64, args.device, srnet, fnet, lr_fnet, net)
        if acc > best_acc:
            best_acc = acc
            save_network_for_backup(args, net, optimizer, scheduler, epoch_id)

    # Save the final SR model
    save_network(args, net, epochs)


def eval():
    fnet, srnet, lr_fnet = initModels()
    if args.Continue:
        net, optimizer, last_epoch, scheduler = backup_init(args)
    else:
        net, optimizer, last_epoch, scheduler = common_init(args)
    acc = val.val_simple_fusion(-1, 8, 16, args.device, srnet, fnet, lr_fnet)


args = train_args.get_args()
if __name__ == '__main__':
    if args.type == 'train':
        main()
    else:
        eval()
