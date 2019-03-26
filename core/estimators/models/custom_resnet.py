import torch.nn as nn
from .resnet import *


class Encoder3x3(ResNetEncoder):
    def __init__(self, in_channel, depths, *args, **kwargs):
        super().__init__(in_channel, depths, *args, **kwargs)

        self.gate = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

class Encoder7x7(ResNetEncoder):
    def __init__(self, in_channel, depths, *args, **kwargs):
        super().__init__(in_channel, depths, *args, **kwargs)

        self.gate = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=7, stride=2, bias=False, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )


class Encoder5x5(ResNetEncoder):
    def __init__(self, in_channel, depths, *args, **kwargs):
        super().__init__(in_channel, depths, *args, **kwargs)

        self.gate = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=5, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

class MicroResnet(ResNet):
    @classmethod
    def micro(cls, in_channel, n=5, *args, **kwargs):
        return cls(in_channel=in_channel, depths=[n, n, n],
                   blocks_sizes=[(16, 32), (32, 64), (64, 128)],
                   n_classes=2, *args, **kwargs)
    @classmethod
    def micro2(cls, in_channel, n=5, *args, **kwargs):
        return cls(in_channel=in_channel, depths=[n, n, n],
                   blocks_sizes=[(8, 16), (16, 32), (32, 64)],
                   n_classes=2, *args, **kwargs)