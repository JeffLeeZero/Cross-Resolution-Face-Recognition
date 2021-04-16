import torch.nn as nn
import torch.nn.functional as F


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.gamma = 1

    def forward(self, classes, target, feature, target_feature):
        loss_class = nn.CrossEntropyLoss()(classes, target).mean()
        loss_feature = F.pairwise_distance(feature, target_feature, p=2).mean()
        loss = loss_feature * self.gamma + loss_class
        return loss, loss_class.item(), loss_feature.item()
