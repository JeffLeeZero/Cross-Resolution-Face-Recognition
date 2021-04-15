import torch.nn as nn
import torch


class SeBlock(nn.Module):
    def __init__(self, channels, r):
        super(SeBlock, self).__init__()
        self.r = r
        self.resblock = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(channels)
        )
        self.pre = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels // r, kernel_size=1)
        )
        self.seblock = nn.Sequential(
            nn.Conv2d(channels // r + 2, channels // r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def load_weight(self, resblock):
        self.resblock = resblock.resblock

    def forward(self, inputs):
        x, down_factor = inputs
        res_output = self.resblock(x)
        w = self.seblock(torch.cat([self.pre(res_output), down_factor], dim=1))
        return x + res_output * w, down_factor
