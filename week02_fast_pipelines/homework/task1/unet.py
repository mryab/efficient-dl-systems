import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down_1 = nn.Sequential(ConvBlock(3, 16), ConvBlock(16, 32, stride=2, padding=1))
        self.down_2 = nn.Sequential(ConvBlock(32, 64), ConvBlock(64, 128))
        self.middle = ConvBlock(128, 128, kernel_size=1, padding=0)
        self.up_2 = nn.Sequential(ConvBlock(256, 128), ConvBlock(128, 32))
        self.up_1 = nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 32))
        self.output = nn.Sequential(ConvBlock(32, 16), ConvBlock(16, 1, kernel_size=1, padding=0))

    def forward(self, x):
        down1 = self.down_1(x)
        out = F.max_pool2d(down1, kernel_size=2, stride=2)

        down2 = self.down_2(out)
        out = F.max_pool2d(down2, kernel_size=2, stride=2)

        out = self.middle(out)

        out = nn.functional.interpolate(out, scale_factor=2)
        out = torch.cat([down2, out], 1)
        out = self.up_2(out)

        out = nn.functional.interpolate(out, scale_factor=2)
        out = torch.cat([down1, out], 1)
        out = self.up_1(out)

        out = nn.functional.interpolate(out, scale_factor=2)

        return self.output(out)
