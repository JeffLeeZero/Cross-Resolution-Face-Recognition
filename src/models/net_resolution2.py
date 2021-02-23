import torch.nn as nn
import torch
import torch.nn.functional as functional

from models import edsr, sface
from util import common


class NetResolution(nn.Module):
    def __init__(self, srnet, sface, feature_count, srnet_freeze=False, fnet_freeze=False):
        super(NetResolution, self).__init__()
        self.srnet = srnet
        if srnet_freeze:
            common.freeze(self.srnet)
        self.convs = nn.Sequential(*list(sface.children())[:-1])
        if fnet_freeze:
            common.freeze(self.convs)
        self.fc_layer = nn.Sequential(
            nn.Linear(512 * 7 * 6 + 2, 512 * 3),
            nn.PReLU(),
            nn.Linear(512 * 3, 512 * 3),
            nn.PReLU(),
            nn.Linear(512 * 3, 512 * 3),
            nn.PReLU(),
            nn.Linear(512 * 3, 512 * 7 * 6),
            nn.PReLU()
        )
        self.raw_layer = nn.Sequential(*list(sface.children())[-1:])

    def forward(self, x, w, h):
        sr_face = self.srnet(x)
        x = functional.interpolate(sr_face, size=(112, 96), mode='bilinear', align_corners=False)
        x = common.tensor2SFTensor(x)
        raw_x = self.convs(x)
        x = torch.cat([raw_x, w, h], dim=1)
        x = self.fc_layer(x)
        x = self.raw_layer(x + raw_x)
        return x, sr_face

    def freeze(self, part):
        if part == "convs":
            common.freeze(self.convs)
        elif part == "srnet":
            common.freeze(self.srnet)

    def unfreeze(self, part):
        if part == "convs":
            common.unfreeze(self.convs)
        elif part == "srnet":
            common.unfreeze(self.srnet)


def get_model():
    srnet = edsr.Edsr()
    fnet = sface.sface()
    net = NetResolution(srnet=srnet, sface=fnet, feature_count=512)
    return net


def get_pretrain_modle(srnet_path=None, fnet_path=None):
    srnet = edsr.Edsr()
    if srnet_path:
        srnet.load_state_dict(torch.load(srnet_path))

    fnet = sface.sface()
    if fnet_path:
        fnet.load_state_dict(torch.load(fnet_path))
    net = NetResolution(srnet=srnet, sface=fnet, feature_count=512)
    return net
