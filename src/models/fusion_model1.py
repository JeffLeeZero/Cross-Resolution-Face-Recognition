import torch.nn as nn
from models.sface_celeba import AngleLinear
import torch

from util import common


class FusionModel(nn.Module):
    def __init__(self, input_size=1024, output_size=512, feature_dim=10178):
        super(FusionModel, self).__init__()
        self.val = False
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.ReLU()
        )
        self.angle = AngleLinear(output_size, feature_dim)

    def forward(self, feature1, feature2):
        x = torch.cat([feature1, feature2], dim=1)
        feature = self.fc(x)
        if self.val:
            return feature, None
        classes = self.angle(feature)
        return feature, classes

    def setVal(self, val):
        self.val = val

        
def getFeatures(srnet, fnet, lr_fnet, lr_face):
    sr_face = srnet(lr_face.clone().detach()).detach()
    # Feature loss
    sr_face_up = nn.functional.interpolate(sr_face, size=(112, 96), mode='bilinear', align_corners=False)
    feature1 = fnet(common.tensor2SFTensor(sr_face_up)).detach()
    lr_face = nn.functional.interpolate(lr_face, size=(112, 96), mode='bilinear', align_corners=False)
    lr_face = common.tensor2SFTensor(lr_face)
    feature2 = lr_fnet(lr_face).detach()
    return feature1, feature2
