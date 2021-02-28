import torch
import torch.nn as nn
import torch.optim as optim
from models import sface_celeba

import os
from tqdm import tqdm
from loaders import celeba_loader
from arguments import train_args
from util import common
from losses.sphere_loss import SphereLoss


def backup_init(args):
    checkpoint = torch.load(args.model_file)
    net = sface_celeba.get_net()
    net.load_state_dict(checkpoint['net'])
    net.to(args.device)
    if len(args.gpu_ids) > 1:
        net = nn.DataParallel(net)
    lr = checkpoint['lr']
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数

    last_epoch = checkpoint['epoch_id']  # 设置开始的epoch
    return net, optimizer, last_epoch, lr


def common_init(args):
    net = sface_celeba.get_net_from_pretrain('../../pretrained/sface.pth')
    net.to(args.device)
    if len(args.gpu_ids) > 1:
        net = nn.DataParallel(net)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    last_epoch = -1
    return net, optimizer, last_epoch, args.lr


def save_network_for_backup(args, srnet, optimizer, epoch_id, lr):
    checkpoint = {
        "net": srnet.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch_id": epoch_id,
        "lr": lr
    }

    save_filename = 'backup_epoch{}.pth'.format(epoch_id)
    save_dir = os.path.join(args.backup_dir, args.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(checkpoint, save_path)


def main():
    args = train_args.get_args()
    dataloader = celeba_loader.get_loader_with_id(args)
    it = iter(dataloader)
    data = next(it)
    if args.Continue:
        net, optimizer, last_epoch, lr = backup_init(args)
    else:
        net, optimizer, last_epoch, lr = common_init(args)
    epoch = args.epoch
    criterion = SphereLoss()
    for epoch_id in range(last_epoch + 1, epoch):
        if epoch_id in [0, 10, 15, 18]:
            if epoch_id != 0: lr *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        bar = tqdm(dataloader, total=len(dataloader), ncols=0)
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, data in enumerate(bar):
            img, label = data
            if img is None: break
            inputs = img.to(args.device)
            targets = label.to(args.device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            lossd = loss.data[0]
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            outputs = outputs[0]  # 0=cos_theta 1=phi_theta
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            description = "epoch {}: mean_loss={} | mean_acc={}% ({}/{}) | loss={}".format(epoch_id,
                                                                                           train_loss / (batch_idx + 1),
                                                                                           100.0 * correct / total,
                                                                                           correct, total,
                                                                                           lossd)
            bar.set_description(desc=description)

        save_network_for_backup(args, net, optimizer, epoch_id)

    common.save_network(args, net, "sface_celeba_epoch{}".format(epoch))


if __name__ == '__main__':
    main()
