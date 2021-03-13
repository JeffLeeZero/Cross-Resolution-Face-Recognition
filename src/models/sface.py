import torch
import torch.nn as nn

from models.sface_celeba import AngleLinear


def Make_layer(block, num_filters, num_of_layer):
    layers = []
    for _ in range(num_of_layer):
        layers.append(block(num_filters))
    return nn.Sequential(*layers)


class SphereResBlock(nn.Module):
    def __init__(self, channels):
        super(SphereResBlock, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(channels)
        )

    def forward(self, x):
        return x + self.resblock(x)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def sface():
    feature_dim = 512
    units = [1, 2, 4, 1]
    filters = [64, 128, 256, 512]
    net_list = []
    for i, (num_units, num_filters) in enumerate(zip(units, filters)):
        if i == 0:
            net_list += [nn.Conv2d(3, 64, 3, 2, 1), nn.PReLU(64)]
        elif i == 1:
            net_list += [nn.Conv2d(64, 128, 3, 2, 1), nn.PReLU(128)]
        elif i == 2:
            net_list += [nn.Conv2d(128, 256, 3, 2, 1), nn.PReLU(256)]
        elif i == 3:
            net_list += [nn.Conv2d(256, 512, 3, 2, 1), nn.PReLU(512)]
        if num_units > 0:
            net_list += [Make_layer(SphereResBlock, num_filters=num_filters, num_of_layer=num_units)]
    net_list += [Flatten()]
    net_list += [nn.Linear(512 * 7 * 6, feature_dim)]
    return nn.Sequential(*net_list)


class SphereFace(nn.Module):
    def __init__(self, type='student', feature_dim=10178, pretrain=None):
        super(SphereFace, self).__init__()
        model = sface()
        if pretrain:
            model.load_state_dict(pretrain)
        if type == 'student':
            self.fc_angle = AngleLinear(512, feature_dim)
        else:
            self.fc_angle = None
        self.convs = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Sequential(*list(model.children())[-1:])
        self.feature = []
        self.val = False

    def forward(self, x):
        #self.feature = self.convs(x)
        #x = self.fc(self.feature)
        x = self.fc(self.convs(x))
        self.feature = x
        if self.fc_angle and not self.val:
            return self.fc_angle(x)
        return x

    def getFeature(self):
        return self.feature

    def setVal(self, val):
        self.val = val