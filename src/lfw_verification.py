import torch
import torch.nn as nn
from models import sface

import os
import numpy as np
from models import fusion_model
from arguments import test_args
from tqdm import tqdm

from loaders import lfw_loader
from util.common import KFold, find_best_threshold, eval_acc, tensor_pair_cosine_distance, \
    tensor_sface_norm, tensors_cvBicubic_resize
import torch.nn.functional as functional


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


def val_sphereface(size, w, h, lfw_bs, device, fnet, net, step=None, index=1):
    net.eval()
    fnet.eval()
    tensor_norm = tensor_sface_norm
    dataloader = lfw_loader.get_loader(size, index, w, h, lfw_bs)
    features11_total, features12_total = [], []
    features21_total, features22_total = [], []
    labels = []
    with torch.no_grad():
        bs_total = 0
        for index, (img1, img2, img1_flip, img2_flip, targets) in enumerate(tqdm(dataloader, ncols=0)):
            bs = len(targets)
            img1, img1_flip = img1.to(device), img1_flip.to(device)
            img2, img2_flip = img2.to(device), img2_flip.to(device)
            img2 = functional.interpolate(img2, size=(112, 96), mode='bilinear', align_corners=False)
            img2_flip = functional.interpolate(img2_flip, size=(112, 96), mode='bilinear', align_corners=False)

            img1, img1_flip = tensor_norm(img1), tensor_norm(img1_flip)
            img2, img2_flip = tensor_norm(img2), tensor_norm(img2_flip)
            features11 = fnet(img1)
            features12 = fnet(img1_flip)
            features21 = net(img2)
            features22 = net(img2_flip)
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

    print(message)
    # if step is not None:
    #     log_name = os.path.join(args.checkpoints_dir, args.name, 'loss_log.txt')
    #     with open(log_name, "a") as log_file:
    #         log_file.write('\n' + message)
    return mean_acc


def val_sesface(size, w, h, lfw_bs, device, fnet, net, step=None, index=1):
    net.eval()
    fnet.eval()
    tensor_norm = tensor_sface_norm
    dataloader = lfw_loader.get_loader(size, index, w, h, lfw_bs)
    features11_total, features12_total = [], []
    features21_total, features22_total = [], []
    labels = []
    with torch.no_grad():
        bs_total = 0
        for i, (img1, img2, img1_flip, img2_flip, targets) in enumerate(tqdm(dataloader, ncols=0)):
            bs = len(targets)
            img1, img1_flip = img1.to(device), img1_flip.to(device)
            img2, img2_flip = img2.to(device), img2_flip.to(device)
            img2 = functional.interpolate(img2, size=(112, 96), mode='bilinear', align_corners=False)
            img2_flip = functional.interpolate(img2_flip, size=(112, 96), mode='bilinear', align_corners=False)

            img1, img1_flip = tensor_norm(img1), tensor_norm(img1_flip)
            img2, img2_flip = tensor_norm(img2), tensor_norm(img2_flip)
            down_factor = torch.ones(size=(bs, 1, 1, 1)).to('cuda:0')
            down_factor *= index / 16
            down_factor2 = 1 / down_factor / 16
            down_factor = torch.cat([down_factor, down_factor2], dim=1)
            fa = torch.ones(size=(bs, 2, 1, 1)).to('cuda:0')
            features11 = net(img1,fa)
            features12 = net(img1_flip,fa)
            features21 = net(img2, down_factor)
            features22 = net(img2_flip, down_factor)
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
        message += '(down_factor:{} {}x{})'.format(index, round(112 / index),
                                                   round(96 / index))
    print(message)
    # if step is not None:
    #     log_name = os.path.join(args.checkpoints_dir, args.name, 'loss_log.txt')
    #     with open(log_name, "a") as log_file:
    #         log_file.write('\n' + message)
    return mean_acc

def val_sesface_self(size, w, h, lfw_bs, device, fnet, net, step=None, index=1):
    net.eval()
    fnet.eval()
    tensor_norm = tensor_sface_norm
    dataloader = lfw_loader.get_loader(size, index, w, h, lfw_bs)
    features11_total, features12_total = [], []
    features21_total, features22_total = [], []
    labels = []
    with torch.no_grad():
        bs_total = 0
        for i, (img1, img2, img1_flip, img2_flip, targets) in enumerate(tqdm(dataloader, ncols=0)):
            bs = len(targets)
            img1, img1_flip = img1.to(device), img1_flip.to(device)
            img2, img2_flip = img2.to(device), img2_flip.to(device)
            img2 = functional.interpolate(img2, size=(112, 96), mode='bilinear', align_corners=False)
            img2_flip = functional.interpolate(img2_flip, size=(112, 96), mode='bilinear', align_corners=False)

            img1, img1_flip = tensor_norm(img1), tensor_norm(img1_flip)
            img2, img2_flip = tensor_norm(img2), tensor_norm(img2_flip)
            #features11 = fnet(img1)
            #features12 = fnet(img1_flip)
            down_factor = torch.ones(size=(bs, 2, 1, 1)).to('cuda:0')
            down_factor[:][0] *= index / 8.0
            down_factor[:][1] *= 1 / index
            features11 = net(img1, down_factor)
            features12 = net(img1_flip, down_factor)
            features21 = net(img2, down_factor)
            features22 = net(img2_flip, down_factor)
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
        message += '(down_factor:{} {}x{})'.format(index, round(112 / index),
                                                   round(96 / index))
    print(message)
    # if step is not None:
    #     log_name = os.path.join(args.checkpoints_dir, args.name, 'loss_log.txt')
    #     with open(log_name, "a") as log_file:
    #         log_file.write('\n' + message)
    return mean_acc

def val_raw(fnet_type, size, down_factor, w, h, lfw_bs, device, fnet, srnet=None, step=None):
    fnet.eval()
    if srnet is not None:
        srnet.eval()
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
            img2, img2_flip = srnet(img2), srnet(img2_flip)
            img2 = tensors_cvBicubic_resize(h, w, device, img2)
            img2_flip = tensors_cvBicubic_resize(h, w, device, img2_flip)
            img1, img1_flip = tensor_norm(img1), tensor_norm(img1_flip)
            img2, img2_flip = tensor_norm(img2), tensor_norm(img2_flip)
            features11 = fnet(img1)
            features12 = fnet(img1_flip)
            features21 = fnet(img2)
            features22 = fnet(img2_flip)
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
    return mean_acc


def val_raw_se(fnet_type, size, down_factor, w, h, lfw_bs, device, fnet, srnet=None, step=None):
    fnet.eval()
    if srnet is not None:
        srnet.eval()
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
            down_factor = torch.ones(size=(bs, 1, 1, 1)).to('cuda:0')
            down_factor *= index / 16
            down_factor2 = 1 / down_factor / 16
            down_f = torch.cat([down_factor, down_factor2], dim=1)
            img2, img2_flip = srnet(img2, down_f), srnet(img2_flip, down_f)
            img2 = tensors_cvBicubic_resize(h, w, device, img2)
            img2_flip = tensors_cvBicubic_resize(h, w, device, img2_flip)
            img1, img1_flip = tensor_norm(img1), tensor_norm(img1_flip)
            img2, img2_flip = tensor_norm(img2), tensor_norm(img2_flip)
            features11 = fnet(img1)
            features12 = fnet(img1_flip)
            features21 = fnet(img2)
            features22 = fnet(img2_flip)
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
    return mean_acc


def get_fusion_feature(srnet, fnet, lr_fnet, net, lr_face, down_f):
    feature1, feature2 = fusion_model.getFeatures(srnet, fnet, lr_fnet, lr_face, down_f)
    feature = net(torch.cat([feature1, feature2], dim=1))
    return feature


def fusion_val(size, down_factor, lfw_bs, device, srnet, fnet, lr_fnet, net=None, step=None):
    net.eval()
    net.setVal(True)
    assert down_factor >= 1, 'Downsampling factor should be >= 1.'
    tensor_norm = tensor_sface_norm
    dataloader = lfw_loader.get_loader(size, down_factor, 96, 112, lfw_bs)
    features11_total, features12_total = [], []
    features21_total, features22_total = [], []
    labels = []
    with torch.no_grad():
        bs_total = 0
        for index, (img1, img2, img1_flip, img2_flip, targets) in enumerate(tqdm(dataloader, ncols=0)):
            bs = len(targets)
            img1, img1_flip = img1.to(device), img1_flip.to(device)
            img2, img2_flip = img2.to(device), img2_flip.to(device)

            down_f = torch.ones(size=(bs, 2, 1, 1)).to('cuda:0')
            down_f[:][0] *= down_factor / 16.0
            down_f[:][1] *= 1 / down_factor
            # img1, img1_flip = tensor_norm(img1), tensor_norm(img1_flip)
            features11 = get_fusion_feature(srnet, fnet, lr_fnet, net, img1, down_f)
            features12 = get_fusion_feature(srnet, fnet, lr_fnet, net, img1_flip, down_f)
            features21 = get_fusion_feature(srnet, fnet, lr_fnet, net, img2, down_f)
            features22 = get_fusion_feature(srnet, fnet, lr_fnet, net, img2_flip, down_f)
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
    net.setVal(False)
    # if step is not None:
    #     log_name = os.path.join(args.checkpoints_dir, args.name, 'loss_log.txt')
    #     with open(log_name, "a") as log_file:
    #         log_file.write('\n' + message)
    return mean_acc


def fusion_val2(size, down_factor, lfw_bs, device, srnet, fnet, lr_fnet, net=None, step=None):
    net.eval()
    net.setVal(True)
    assert down_factor >= 1, 'Downsampling factor should be >= 1.'
    tensor_norm = tensor_sface_norm
    dataloader = lfw_loader.get_loader_features(size, down_factor, 96, 112, lfw_bs)
    features11_total, features12_total = [], []
    features21_total, features22_total = [], []
    labels = []
    with torch.no_grad():
        bs_total = 0
        for index, (features11, img2, features12, img2_flip, targets) in enumerate(tqdm(dataloader, ncols=0)):
            bs = len(targets)
            features11, features12 = features11.to(device), features12.to(device)
            img2, img2_flip = img2.to(device), img2_flip.to(device)

            down_f = torch.ones(size=(bs, 2, 1, 1)).to('cuda:0')
            down_f[:][0] *= down_factor / 16.0
            down_f[:][1] *= 1 / down_factor
            # img1, img1_flip = tensor_norm(img1), tensor_norm(img1_flip)
            features11 = net(features11)
            features12 = net(features12)
            features21 = get_fusion_feature(srnet, fnet, lr_fnet, net, img2, down_f)
            features22 = get_fusion_feature(srnet, fnet, lr_fnet, net, img2_flip, down_f)
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
    net.setVal(False)
    # if step is not None:
    #     log_name = os.path.join(args.checkpoints_dir, args.name, 'loss_log.txt')
    #     with open(log_name, "a") as log_file:
    #         log_file.write('\n' + message)
    return mean_acc





if __name__ == '__main__':
    args = test_args.get_args()

