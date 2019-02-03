import torch.nn as nn
from models.resnet import *


class TraversabilityResNetEncoder(ResNetEncoder):
    def __init__(self, in_channel, blocks, block=BasicBlock, conv_layer=nn.Conv2d, *args, **kwargs):
        super().__init__(in_channel, blocks, block=BasicBlock, conv_layer=nn.Conv2d, *args, **kwargs)

        self.layers[-1] = ResNetLayer(256, 512, depth=blocks[3], block=BasicBlockSE, conv_layer=conv_layer, *args, **kwargs)


        self.initialise(self.modules())

class TraversabilityResNet(ResNet):
    def __init__(self,in_channel, blocks, block=BasicBlock, conv_layer=nn.Conv2d, n_classes=1000, *args, **kwargs):
        super().__init__(in_channel, blocks, block=BasicBlock, conv_layer=nn.Conv2d, n_classes=1000, *args, **kwargs)
        self.encoder = TraversabilityResNetEncoder(in_channel, blocks, block=BasicBlock, conv_layer=nn.Conv2d,  preactivated=True, *args, **kwargs)

    @classmethod
    def resnet34(cls, in_channel, *args, **kwargs):
        return cls(in_channel, blocks=[3, 4, 6, 3], *args, **kwargs)

class MicroResnet(ResNet):

    @classmethod
    def micro(cls, in_channel, *args, **kwargs):
        return cls(in_channel, blocks=[1, 1, 1, 1], *args, **kwargs)
