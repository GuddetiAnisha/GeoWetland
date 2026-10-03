import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels=10):
        super().__init__()
        self.c1 = DoubleConv(in_channels, 32)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = DoubleConv(32, 64)
        self.p2 = nn.MaxPool2d(2)
        self.mid = DoubleConv(64, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d2 = DoubleConv(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        a = self.c1(x)
        b = self.c2(self.p1(a))
        m = self.mid(self.p2(b))
        x = self.u2(m)
        x = self.d2(torch.cat([x, b], 1))
        x = self.u1(x)
        x = self.d1(torch.cat([x, a], 1))
        return self.out(x)
