import torch.nn as nn
import torch.nn.functional as F
from losses.sphere_loss import SphereLoss


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.gamma = 1
        self.sphere_loss = SphereLoss()
        self.l2_loss = nn.MSELoss()

    def forward(self, classes, target, feature, target_feature):
        loss_class = self.sphere_loss(classes, target)
        loss_feature = self.l2_loss(feature, target_feature)
        loss_feature = loss_feature.mean()
        loss = loss_class + loss_feature * self.gamma
        return loss.mean(), loss_class.item(), loss_feature.item()