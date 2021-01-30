from torch.utils.data import DataLoader

from util import common
import torch
import cv2
import pandas as pd

LFW_ROOT = '../datasets/lfw/'
LFW_LANDMARKS = '../data/LFW.csv'
LFW_PAIRS = '../data/lfw_pairs.txt'
CELEBA_ROOT = '../../Datasets/CelebA/img_celeba/'
CELEBA_CSV = '../data/celeba_clean_landmarks.csv'
CELEBA_ID = '../data/identity_CelebA.txt'

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self):
        super(CelebADataset, self).__init__()
        df = pd.read_csv(CELEBA_CSV, delimiter=",")
        self.faces_path = df.values[:, 0]
        self.landmarks = df.values[:, 1:]
        with open(CELEBA_ID) as f:
            id = f.readlines()
        self.id = id

    def __getitem__(self, index):
        path = self.faces_path[index]
        img = cv2.imread(CELEBA_ROOT + path)
        face = common.alignment(img, self.landmarks[index].reshape(-1, 2))
        index = int(path[0:-4])
        id = int(self.id[index].split()[1][0:-1])

        return face, id

    def __len__(self):
        return len(self.faces_path)


def get_loader(args, num_workers=1):
    dataset = CelebADataset()
    dataloader = DataLoader(dataset=dataset,
                            num_workers=num_workers,
                            batch_size=args.bs,
                            shuffle=True,
                            drop_last=True)
    return dataloader
