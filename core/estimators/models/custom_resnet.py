import torch.nn as nn
from models.resnet import *


# class MicroEncoder(ResNetEncoder):
#     @property
#     def blocks_sizes(self):
#         return [(32, 32), (32, 64), (64, 128), (128, 256)]

class MicroResnet(ResNet):
    def __init__(self, in_channel, blocks, block=BasicBlock, conv_layer=nn.Conv2d, n_classes=1000, *args, **kwargs):
        super().__init__(in_channel, blocks, block=block, conv_layer=nn.Conv2d, n_classes=n_classes, *args, **kwargs)

        # self.encoder = MicroEncoder(in_channel, blocks, block=block, conv_layer=conv_layer, *args, **kwargs)
        self.encoder.gate = nn.Sequential(
            conv_layer(in_channel, 64, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    @classmethod
    def micro(cls, in_channel, *args, **kwargs):
        return cls(in_channel, blocks=[1, 1, 1, 1], *args, **kwargs)


# micro-resnet#1 (32, 32), (32, 64), (64, 128), (128, 256)
# micro-resnet#2 like resnet but with gate 5x5