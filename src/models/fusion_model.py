import torch.nn as nn
import torch

from models.angle_linear import AngleLinear
from losses.arcface import ArcFace
from util import common


class FusionModel(nn.Module):
    def __init__(self, input_size=1024, output_size=512, feature_dim=10178):
        super(FusionModel, self).__init__()
        self.val = False
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 512 * 3),
            nn.PReLU(512 * 3),
            nn.Linear(512 * 3, output_size),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.Dropout(0.5)
        )
        self.arcface = ArcFace(output_size, feature_dim)

    def forward(self, x, target=None):
        feature = self.fc1(x)
        if self.val:
            return self.fc2(feature)
        return feature, self.arcface(self.fc2(feature), target)

    def setVal(self, val):
        self.val = val


def getFeatures(srnet, fnet, lr_fnet, lr_face, factor):
    if srnet:
        sr_face = srnet(lr_face.clone().detach(), factor.detach()).detach()
    # Feature loss
        sr_face_up = nn.functional.interpolate(sr_face, size=(112, 96), mode='bilinear', align_corners=False)
        feature1 = fnet(common.tensor2SFTensor(sr_face_up)).detach()
    else:
        feature1 = fnet(lr_face).detach()
    lr_face = nn.functional.interpolate(lr_face, size=(112, 96), mode='bilinear', align_corners=False)
    lr_face = common.tensor2SFTensor(lr_face)
    feature2 = lr_fnet(lr_face, factor.detach()).detach()
    return feature1, feature2
