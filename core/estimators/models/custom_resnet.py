import torch.nn as nn
from models.resnet import *

from torchsummary import summary

# class MicroEncoder(ResNetEncoder):
#     @property
#     def blocks_sizes(self):
#         return [(16, 32), (32, 64), (64, 128)]
#
# class MicroResnet(ResNet):
#     def __init__(self, in_channel, blocks, block=BasicBlock, conv_layer=nn.Conv2d, n_classes=1000, *args, **kwargs):
#         super().__init__(in_channel, blocks, block=block, conv_layer=nn.Conv2d, n_classes=n_classes, decoder=MicroEncoder, *args, **kwargs)
#
#         self.encoder.gate = nn.Sequential(
#             conv_layer(in_channel, 16, kernel_size=5, stride=2, bias=False, padding=1),
#             nn.BatchNorm2d(16),
#             nn.LeakyReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=1)
#         )
#
#     @classmethod
#     def micro(cls, in_channel, n=5, *args, **kwargs):
#         return cls(in_channel, blocks=[1 + n, n, n], *args, **kwargs)
#

# micro-resnet#1 (32, 32), (32, 64), (64, 128), (128, 256)
# micro-resnet#2 like resnet but with gate 5x5
# micro-resnet#3 [(16, 32), (32, 64), (64, 128)] with [1 + n, n, n]

# class ResnetDecoderRegression(ResnetDecoder):
#
#     def __init__(self, in_features, n_classes):
#         super().__init__(in_features, n_classes)
#         self.avg = nn.AdaptiveAvgPool2d((1, 1))
#         self.decoder = nn.Sequential(
#             nn.Linear(in_features, 1),
#             nn.Sigmoid())

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


# from estimators.models.omar_cnn import OmarCNN
#
# summary(OmarCNN(), (1, 92, 92))


# model = MicroResnet.micro(1, n=3, preactivate=True, blocks=[BasicBlock, BasicBlock, BasicBlockSE])
#
# summary(model, (1, 92, 92))
