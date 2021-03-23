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


class FusionLoss3(nn.Module):
    def __init__(self):
        super(FusionLoss3, self).__init__()
        self.gamma = 1
        self.sphere_loss = SphereLoss()

    def forward(self, classes, target, feature, target_feature):
        loss_class = 0  # self.sphere_loss(classes, target)
        loss_feature = F.pairwise_distance(feature, target_feature, p=2).mean()
        loss_feature = loss_feature.mean()
        loss_cos = 1 - nn.CosineSimilarity()(feature, target_feature)
        loss_cos = loss_cos.mean()
        loss = loss_feature * self.gamma + loss_cos * 20
        return loss.mean(), loss_class, loss_feature.item(), loss_cos.item()
