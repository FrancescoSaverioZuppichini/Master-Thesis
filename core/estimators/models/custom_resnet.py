import torch.nn as nn
from models.resnet import *


class MicroEncoder(ResNetEncoder):
    @property
    def blocks_sizes(self):
        return [(16, 32), (32, 64), (64, 128)]

class MicroResnet(ResNet):
    def __init__(self, in_channel, blocks, block=BasicBlock, conv_layer=nn.Conv2d, n_classes=1000, *args, **kwargs):
        super().__init__(in_channel, blocks, block=block, conv_layer=nn.Conv2d, n_classes=n_classes, decoder=MicroEncoder, *args, **kwargs)

        self.encoder.gate = nn.Sequential(
            conv_layer(in_channel, 16, kernel_size=5, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

    @classmethod
    def micro(cls, in_channel, n=5, *args, **kwargs):
        return cls(in_channel, blocks=[1 + n, n, n], *args, **kwargs)


# micro-resnet#1 (32, 32), (32, 64), (64, 128), (128, 256)
# micro-resnet#2 like resnet but with gate 5x5
# micro-resnet#3 [(16, 32), (32, 64), (64, 128)] with [1 + n, n, n]