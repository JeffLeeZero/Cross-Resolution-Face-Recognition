from torch import nn


class Demo(nn.Module):
    def __init__(self):
        super(Demo).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features=2, out_features=9),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer(x)
        return x.view(3, 3, -1)

