from torch.utils.data import DataLoader

from util import common
import torch
import cv2
import pandas as pd
import random

CELEBA_ROOT = '../../img_align_celeba/'
CELEBA_CSV = '../data/celeba_clean_landmarks.csv'
CELEBA_ID = '../data/identity_CelebA.txt'


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self):
        super(CelebADataset, self).__init__()
        df = pd.read_csv(CELEBA_CSV, delimiter=",")
        self.faces_path = df.values[:, 0]
        self.landmarks = df.values[:, 1:]
        self.id = {}
        with open(CELEBA_ID) as f:
            ids = f.readlines()
        for item in ids:
            id = item.split()
            self.id[id[0]] = id[1].split('"')[0]

    def __getitem__(self, index):
        path = self.faces_path[index]
        img = cv2.imread(CELEBA_ROOT + path)
        face = common.alignment(img, self.landmarks[index].reshape(-1, 2))
        id = int(self.id[path])

        return common.face_ToTensor(face), id

    def __len__(self):
        return len(self.faces_path)


class CelebADatasetDownsample(torch.utils.data.Dataset):
    def __init__(self):
        super(CelebADatasetDownsample, self).__init__()
        df = pd.read_csv(CELEBA_CSV, delimiter=",")
        self.faces_path = df.values[:, 0]
        self.landmarks = df.values[:, 1:]
        self.id = {}
        with open(CELEBA_ID) as f:
            ids = f.readlines()
        for item in ids:
            id = item.split()
            self.id[id[0]] = id[1].split('"')[0]

    def __getitem__(self, index):
        path = self.faces_path[index]
        img = cv2.imread(CELEBA_ROOT + path)
        face = common.alignment(img, self.landmarks[index].reshape(-1, 2))

        if random.random() > 0.5:
            face = cv2.flip(face, 1)
        face_down2 = cv2.resize(face, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_CUBIC)
        face_down4 = cv2.resize(face, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_CUBIC)
        face_down8 = cv2.resize(face, None, fx=1 / 8, fy=1 / 8, interpolation=cv2.INTER_CUBIC)
        face_down16 = cv2.resize(face, None, fx=1 / 16, fy=1 / 16, interpolation=cv2.INTER_CUBIC)
        id = int(self.id[path])
        face_dict = {'down1': common.face_ToTensor(face),
                     'down2': common.face_ToTensor(face_down2),
                     'down4': common.face_ToTensor(face_down4),
                     'down8': common.face_ToTensor(face_down8),
                     'down16': common.face_ToTensor(face_down16),
                     'id': id}

        return face_dict

    def __len__(self):
        return len(self.faces_path)


def get_loader_with_id(args, num_workers=1):
    dataset = CelebADataset()
    dataloader = DataLoader(dataset=dataset,
                            num_workers=num_workers,
                            batch_size=args.bs,
                            shuffle=True,
                            drop_last=True)
    return dataloader


def get_loader_downsample(args, num_workers=1):
    dataset = CelebADatasetDownsample()
    dataloader = DataLoader(dataset=dataset,
                            num_workers=num_workers,
                            batch_size=args.bs,
                            shuffle=True,
                            drop_last=True)
    return dataloader
