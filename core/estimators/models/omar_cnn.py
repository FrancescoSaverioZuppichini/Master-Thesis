import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=3,
                                   padding=1,
                                   bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())


class OmarCNN(nn.Module):
    """
    From original paper https://arxiv.org/abs/1709.05368
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(conv_block(1, 5),
                                     conv_block(5, 5),
                                     nn.MaxPool2d(kernel_size=2),
                                     conv_block(5, 5))

        self.decoder = nn.Sequential(nn.Linear(1280, 128),
                                     nn.ReLU(),
                                     nn.Dropout(),
                                     nn.Linear(128, 2))

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)

        return x
