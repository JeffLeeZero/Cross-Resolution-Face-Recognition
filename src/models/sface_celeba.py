from models import sface
import torch
import torch.nn as nn
from util import common
from . import sface


class SfaceCelebA(nn.Module):
    def __init__(self, sface, feature_count=10177):
        super(SfaceCelebA, self).__init__()
        self.fnet_without_fc = nn.Sequential(*list(sface.children())[:-1])
        common.freeze(self.fnet_without_fc)
        self.fc_layer = nn.Linear(512 * 7 * 6, feature_count)

    def forward(self, x):
        x = self.fnet_without_fc(x)
        x = self.fc_layer(x)
        return x


def get_net():
    pretrain_net = sface.sface()
    net = SfaceCelebA(pretrain_net)
    return net


def get_net_from_pretrain(sface_path):
    pretrain_net = sface.sface()
    pretrain_net.load_state_dict(torch.load(sface_path))
    net = SfaceCelebA(pretrain_net)
    return net
