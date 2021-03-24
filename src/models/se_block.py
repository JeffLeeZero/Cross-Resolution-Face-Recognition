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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.seblock = nn.Sequential(
            nn.Conv2d(channels + 2, channels // r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r , channels, kernel_size=1),
            nn.Sigmoid()
        )

    def load_weight(self, resblock):
        self.resblock = resblock.resblock

    def forward(self, x, down_factor):
        res_output = self.resblock(x)

        return x + res_output * self.seblock(torch.cat([self.pool(res_output), down_factor], dim=1))
