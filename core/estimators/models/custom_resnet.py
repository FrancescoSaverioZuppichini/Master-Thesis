import torch.nn as nn
from models.resnet import *

class MicroResnet(ResNet):
    def __init__(self, depths, in_channel,  *args, **kwargs):
        super().__init__(depths, in_channel, *args, **kwargs)

        self.encoder.gate = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    @classmethod
    def micro(cls, in_channel, n=5, *args, **kwargs):
        return cls(in_channel=in_channel, depths=[n, n, n],
                   blocks_sizes=[(16, 32), (32, 64), (64, 128)],
                   n_classes=2, *args, **kwargs)
