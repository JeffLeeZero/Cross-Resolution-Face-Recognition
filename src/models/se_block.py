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
        self.seblock1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels // r, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.seblock2 = nn.Sequential(
            nn.Conv2d(channels // r + 1, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def load_weight(self, resblock):
        self.resblock = resblock.resblock

    def forward(self, x, down_factor):
        res_output = self.resblock(x)

        return x + res_output * (self.seblock2(torch.cat([self.seblock1(res_output), down_factor], dim=1))), down_factor
