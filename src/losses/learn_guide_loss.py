import torch.nn as nn
import torch.nn.functional as F

class LearnGuideLoss(nn.Module):
    def __init__(self):
        super(LearnGuideLoss, self).__init__()
        self.gamma = 1

    def forward(self, classes, target, lr_feature, hr_feature):
        loss_class = nn.CrossEntropyLoss()(classes, target).mean()
        loss_feature = F.pairwise_distance(lr_feature, hr_feature, p=2).mean()
        loss = loss_class + loss_feature * self.gamma
        return loss, loss_class.item(), loss_feature.item()
