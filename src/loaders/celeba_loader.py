from util import common
import torch
import cv2
import pandas as pd

LFW_ROOT = '../datasets/lfw/'
LFW_LANDMARKS = '../data/LFW.csv'
LFW_PAIRS = '../data/lfw_pairs.txt'
CELEBA_ROOT = '../../../Datasets/CelebA/img_celeba/'
CELEBA_CSV = '../data/celeba_clean_landmarks.csv'


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self):
        super(CelebADataset, self).__init__()
        df = pd.read_csv(CELEBA_CSV, delimiter=",")
        self.faces_path = df.values[:, 0]
        self.landmarks = df.values[:, 1:]
        self.id = []

    def __getitem__(self, index):
        img = cv2.imread(CELEBA_ROOT + self.faces_path[index])
        face = common.alignment(img, self.landmarks[index].reshape(-1, 2))
        id = self.id[index]

        return face, id

    def __len__(self):
        return len(self.faces_path)
