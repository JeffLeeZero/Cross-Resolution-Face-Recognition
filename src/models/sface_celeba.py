import torch
import torch.nn as nn

from models.angle_linear import AngleLinear
from util import common


class SfaceCelebA(nn.Module):
    def __init__(self, sface, feature_count=10178):
        super(SfaceCelebA, self).__init__()
        self.val = False
        self.fnet_without_fc = nn.Sequential(*list(sface.children())[:-1])
        common.freeze(self.fnet_without_fc)
        self.fc_layer = nn.Linear(512 * 7 * 6, 512)
        self.fc_angle = AngleLinear(512, feature_count)

    def forward(self, x):
        x = self.fnet_without_fc(x)
        x = self.fc_layer(x)
        if self.val:
            return x
        return self.fc_angle(x)

    def setVal(self, val):
        self.val = val


def get_net():
    pretrain_net = sface()
    net = SfaceCelebA(pretrain_net)
    return net


def get_net_from_pretrain(sface_path):
    pretrain_net = sface()
    pretrain_net.load_state_dict(torch.load(sface_path))
    net = SfaceCelebA(pretrain_net)
    return net
