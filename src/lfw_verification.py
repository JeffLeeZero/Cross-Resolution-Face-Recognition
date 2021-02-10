import torch
import torch.nn as nn
from models import sface, net_resolution

import os
import numpy as np

from arguments import test_args
from tqdm import tqdm

from loaders import lfw_loader
from util.common import KFold, find_best_threshold, eval_acc, tensor_pair_cosine_distance, \
    tensor_sface_norm, tensors_cvBicubic_resize


def run(fnet_type, size, down_factor, w, h, lfw_bs, device, fnet, net=None, step=None):
    fnet.eval()
    if net is not None:
        net.eval()
    assert down_factor >= 1, 'Downsampling factor should be >= 1.'
    if fnet_type == 'sface':
        tensor_norm = tensor_sface_norm
    dataloader = lfw_loader.get_loader(size, down_factor, w, h, lfw_bs)
    features11_total, features12_total = [], []
    features21_total, features22_total = [], []
    labels = []
    with torch.no_grad():
        bs_total = 0
        for index, (img1, img2, img1_flip, img2_flip, targets) in enumerate(tqdm(dataloader, ncols=0)):
            bs = len(targets)
            img1, img1_flip = img1.to(device), img1_flip.to(device)
            img2, img2_flip = img2.to(device), img2_flip.to(device)
            img1, img1_flip = tensor_norm(img1), tensor_norm(img1_flip)
            features11 = fnet(img1)
            features12 = fnet(img1_flip)
            w = torch.full([img2.shape[0], 1], img2.shape[2]).to(device)
            h = torch.full([img2.shape[0], 1], img2.shape[3]).to(device)
            features21, _ = net(img2, w, h)
            w = torch.full([img2_flip.shape[0], 1], img2_flip.shape[2]).to(device)
            h = torch.full([img2_flip.shape[0], 1], img2_flip.shape[3]).to(device)
            features22, _ = net(img2_flip, w, h)
            features11_total += [features11]
            features12_total += [features12]
            features21_total += [features21]
            features22_total += [features22]
            labels += [targets]
            bs_total += bs
        features11_total = torch.cat(features11_total)
        features12_total = torch.cat(features12_total)
        features21_total = torch.cat(features21_total)
        features22_total = torch.cat(features22_total)
        labels = torch.cat(labels)
        assert bs_total == 6000, print('LFW pairs should be 6,000')
    labels = labels.cpu().numpy()
    scores = tensor_pair_cosine_distance(features11_total, features12_total, features21_total, features22_total,
                                         type='concat')
    accuracy = []
    thd = []
    folds = KFold(n=6000, n_folds=10, shuffle=False)
    thresholds = np.linspace(-10000, 10000, 10000 + 1)
    thresholds = thresholds / 10000
    predicts = np.hstack((scores, labels))
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    mean_acc, std = np.mean(accuracy), np.std(accuracy)
    if step is not None:
        message = 'LFWACC={:.4f} std={:.4f} at {}iter.'.format(mean_acc, std, step)
    else:
        message = 'LFWACC={:.4f} std={:.4f} at testing.'.format(mean_acc, std)
    if size != -1:
        message += '(down_factor: {}x{})'.format(size, size)
    else:
        message += '(down_factor:{} {}x{})'.format(down_factor, round(112 / down_factor),
                                                   round(96 / down_factor))

    print(message)
    # if step is not None:
    #     log_name = os.path.join(args.checkpoints_dir, args.name, 'loss_log.txt')
    #     with open(log_name, "a") as log_file:
    #         log_file.write('\n' + message)
    return mean_acc


if __name__ == '__main__':
    args = test_args.get_args()
    print('----------------- Start -----------------')
    ## FNet
    fnet = sface.sface()
    fnet.load_state_dict(torch.load(args.fnet_pth))
    if args.fnet == 'sface':
        args.w, args.h = 96, 112
        args.feature_dim = 512
    fnet.to(args.device)
    if len(args.gpu_ids) > 1:
        fnet = nn.DataParallel(fnet)
    print('===> FNet: {} is used.'.format(args.fnet))
    print('===> {} is loaded.'.format(args.fnet_pth))
    ## SRNet
    if args.isSR:
        net = net_resolution.get_model()
        if args.Continue:
            checkpoint = torch.load(args.srnet_pth)
            net.load_state_dict(checkpoint['net'])
        else:
            net.load_state_dict(torch.load(args.srnet_pth))

        net.to(args.device)
        if len(args.gpu_ids) > 1:
            srnet = nn.DataParallel(net)
        print('===> {} is loaded.'.format(args.srnet_pth))
    else:
        srnet = None
        print('===> Bicubic interpolation is used.')
    ## LR face verification
    run(args.fnet, args.size, args.down_factor, args.w, args.h, args.lfw_bs, args.device, fnet, net)
    print('------------------ End ------------------ ')
    print('')
