import torch
from torch.utils.data import DataLoader

import cv2
import numpy as np
import pandas as pd
from util.common import alignment, face_ToTensor

LFW_ROOT = '../../Datasets/lfw/'
LFW_LANDMARKS = '../data/LFW.csv'
LFW_PAIRS = '../data/lfw_pairs.txt'
LFW_FEATURES = '../../Datasets/lfw_features.pth'
CELEBA_ROOT = '../../Datasets/CelebA/img_celeba/'
CELEBA_CSV = '../data/celeba_clean_landmarks.csv'
CELEBA_ID = '../data/identity_CelebA.txt'


class LFWDataset(torch.utils.data.Dataset):
    def __init__(self, size, down_factor, w, h, isSR=True):
        super(LFWDataset, self).__init__()
        df = pd.read_csv(LFW_LANDMARKS, delimiter=",", header=None)
        numpyMatrix = df.values
        self.landmarks = numpyMatrix[:, 1:]
        self.df = df
        self.size = size
        self.down_factor = down_factor
        self.w = w
        self.h = h
        self.isSR = isSR
        with open(LFW_PAIRS) as f:
            pairs_lines = f.readlines()[1:]
        self.pairs_lines = pairs_lines

    def __getitem__(self, index):
        p = self.pairs_lines[index].replace('\n', '').split('\t')
        if 3 == len(p):
            sameflag = np.int32(1).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        if 4 == len(p):
            sameflag = np.int32(0).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        img1 = alignment(cv2.imread(LFW_ROOT + name1),
                         self.landmarks[self.df.loc[self.df[0] == name1].index.values[0]])
        img2 = alignment(cv2.imread(LFW_ROOT + name2),
                         self.landmarks[self.df.loc[self.df[0] == name2].index.values[0]])
        ## Resize second image
        if self.size != -1:
            ## Use args.size
            img2 = cv2.resize(img2, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        else:
            ## Use args.down_factor
            img2 = cv2.resize(img2, None, fx=1 / self.down_factor, fy=1 / self.down_factor,
                              interpolation=cv2.INTER_CUBIC)

        ## Resize the to the required size of FNet
        img1 = cv2.resize(img1, (self.w, self.h), cv2.INTER_CUBIC)
        if not self.isSR:
            img2 = cv2.resize(img2, (self.w, self.h), cv2.INTER_CUBIC)

        ## Obtain the mirror faces
        img1_flip = cv2.flip(img1, 1)
        img2_flip = cv2.flip(img2, 1)

        return face_ToTensor(img1), face_ToTensor(img2), \
               face_ToTensor(img1_flip), face_ToTensor(img2_flip), \
               torch.LongTensor(sameflag)

    def __len__(self):
        return len(self.pairs_lines)


class LFWDatasetsFeatures(torch.utils.data.Dataset):
    def __init__(self,size, down_factor, w, h, isSR=True):
        super(LFWDatasetsFeatures, self).__init__()
        self.features = torch.load(LFW_FEATURES)

        df = pd.read_csv(LFW_LANDMARKS, delimiter=",", header=None)
        numpyMatrix = df.values
        self.landmarks = numpyMatrix[:, 1:]
        self.df = df
        self.size = size
        self.down_factor = down_factor
        self.w = w
        self.h = h
        self.isSR = isSR
        with open(LFW_PAIRS) as f:
            pairs_lines = f.readlines()[1:]
        self.pairs_lines = pairs_lines
        # df = pd.read_csv(CELEBA_CSV, delimiter=",")
        # self.faces_path = df.values[:, 0]
        # self.landmarks = df.values[:, 1:]

    def __getitem__(self, index):
        p = self.pairs_lines[index].replace('\n', '').split('\t')
        if 3 == len(p):
            sameflag = np.int32(1).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        if 4 == len(p):
            sameflag = np.int32(0).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        img1 = alignment(cv2.imread(LFW_ROOT + name1),
                         self.landmarks[self.df.loc[self.df[0] == name1].index.values[0]])
        img2 = alignment(cv2.imread(LFW_ROOT + name2),
                         self.landmarks[self.df.loc[self.df[0] == name2].index.values[0]])
        ## Resize second image
        if self.size != -1:
            ## Use args.size
            img2 = cv2.resize(img2, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        else:
            ## Use args.down_factor
            img2 = cv2.resize(img2, None, fx=1 / self.down_factor, fy=1 / self.down_factor,
                              interpolation=cv2.INTER_CUBIC)

        ## Resize the to the required size of FNet
        img1 = cv2.resize(img1, (self.w, self.h), cv2.INTER_CUBIC)
        if not self.isSR:
            img2 = cv2.resize(img2, (self.w, self.h), cv2.INTER_CUBIC)

        ## Obtain the mirror faces
        img1_flip = cv2.flip(img1, 1)
        img2_flip = cv2.flip(img2, 1)

        return self.features[index][0:1024], face_ToTensor(img2), \
               self.features[index][1024:2048], face_ToTensor(img2_flip), \
               torch.LongTensor(sameflag)

    def __len__(self):
        return len(self.pairs_lines)


def get_loader(size, down_factor, w, h, lfw_bs, num_workers=1, isSR=True):
    dataset = LFWDataset(size, down_factor, w, h,isSR)
    dataloader = DataLoader(dataset=dataset,
                            num_workers=num_workers,
                            batch_size=lfw_bs,
                            shuffle=False,
                            drop_last=False)
    return dataloader

def get_loader_features(size, down_factor, w, h, lfw_bs, num_workers=1, isSR=True):
    dataset = LFWDatasetsFeatures(size, down_factor, w, h,isSR)
    dataloader = DataLoader(dataset=dataset,
                            num_workers=num_workers,
                            batch_size=lfw_bs,
                            shuffle=False,
                            drop_last=False)
    return dataloader