import torch.nn as nn
import torch.nn.functional as F
from losses.sphere_loss import SphereLoss


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.gamma = 1
        self.sphere_loss = SphereLoss()

    def forward(self, classes, target, feature, target_feature):
        loss_class = self.sphere_loss(classes, target)
        loss_feature = F.pairwise_distance(feature, target_feature, p=2).mean()
        loss_feature = loss_feature.mean()
        loss = loss_class + loss_feature * self.gamma
        return loss.mean(), loss_class.item(), loss_feature.item()


class FusionLoss2(nn.Module):
    def __init__(self):
        super(FusionLoss2, self).__init__()
        self.sphere_loss = SphereLoss()

    def forward(self, classes, target):
        return self.sphere_loss(classes, target)