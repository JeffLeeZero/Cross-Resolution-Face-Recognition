from models import sface
import torch
import torch.nn as nn
from util import common


class FNet1(nn.Module):
    def __init__(self, sface, feature_count):
        super(FNet1, self).__init__()
        self.fnet_without_fc = nn.Sequential(*list(sface.children())[:-1])
        common.freeze(self.fnet_without_fc)
        self.fc_layer = nn.Linear(512 * 7 * 6, feature_count)

    def forward(self, x):
        x = self.fnet_without_fc(x)
        x = self.fc_layer(x)
        return x
