import torch.nn as nn
import torch.nn.functional as F
from losses.sphere_loss import SphereLoss


class LearnGuideLoss(nn.Module):
    def __init__(self):
        super(LearnGuideLoss, self).__init__()
        self.gamma = 0.5
        self.sphere_loss = SphereLoss()

    def forward(self, classes, target, lr_feature, hr_feature):
        loss_class = self.sphere_loss(classes, target)
        loss_feature = F.pairwise_distance(lr_feature, hr_feature, p=2)
        loss = loss_class + loss_feature * self.gamma
        return loss, loss_class.float(), loss_feature.float()
