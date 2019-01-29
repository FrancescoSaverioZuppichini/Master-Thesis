import torch
import torch.nn as nn
# from torchvision.models import resnet34, resnet18, ResNet
# from torchvision.models.resnet import ResNet, BasicBlock


def tail(type, in_features=512, out_features=2):
    return nn.ModuleDict({
        'short': nn.Linear(in_features, out_features),
        'long': nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features//2, out_features)
        )
    })[type]

class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, stride=1, *args, **kwargs):
        super().__init__()
        padding = ((in_channels - 1) * (stride - 1) + 1 * (kernel_size - 1)) // 2

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, padding=padding)
        )
    def forward(self, x):
        return self.net(x)

class SEModule(nn.Module):
    def __init__(self, n_features, ratio=16, *args, **kwargs):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Linear(n_features, n_features // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(n_features // ratio, n_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)  # flat
        out = self.se(out).view(b, c, 1, 1)

        return x * out

def conv_block(in_planes, out_planes, conv_layer=nn.Conv2d, kernel_size=3, padding=None, preactivated=False, stride=1, **kwargs):
    padding = kernel_size//2  if not padding else padding

    if preactivated:
        conv_block =  nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.LeakyReLU(negative_slope=0.1),
            conv_layer(in_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False, stride=stride, **kwargs)
        )
    else:
        conv_block = nn.Sequential(
            conv_layer(in_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False, stride=stride, **kwargs),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(),
        )

    return conv_block

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, conv_layer=nn.Conv2d, *args, **kwargs):
        super().__init__()

        self.shortcut = None


        self.block = nn.Sequential(
            conv_block(in_planes, out_planes, conv_layer, stride=stride, *args, **kwargs),
            conv_block(out_planes, out_planes, conv_layer, *args, **kwargs),
        )

        if in_planes != out_planes:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, out_planes, kernel_size=1,
                                      stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        residual = x

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out = self.block(x)

        out = out + residual

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, conv_layer=nn.Conv2d, *args, **kwargs):
        super().__init__()

        assert in_planes % self.expansion == 0

        self.shortcut = None
        self.expanded = out_planes * self.expansion

        self.block = nn.Sequential(
            conv_block(in_planes, out_planes, conv_layer, kernel_size=1),
            conv_block(out_planes, out_planes, conv_layer, kernel_size=3, stride=stride),
            conv_block(out_planes, self.expanded, conv_layer, kernel_size=1),
        )

        if in_planes !=  self.expanded:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, self.expanded, kernel_size=1,
                           stride=stride, bias=False),
                nn.BatchNorm2d(self.expanded),
                )

    def forward(self, x):
        residual = x

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out = self.block(x)

        out.add_(residual)

        return out

class BasicBlockSE(BasicBlock):
    def __init__(self, in_planes, out_planes, conv_layer=nn.Conv2d, *args, **kwargs):
        super().__init__(in_planes, out_planes, conv_layer=conv_layer, *args, **kwargs)
        self.se = SEModule(out_planes)

    def forward(self, x):
        residual = x

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out = self.block(x)
        out = self.se(out )
        out += residual

        return out

class BottleneckSE(Bottleneck):
    expansion = 4

    def __init__(self, in_planes, out_planes, *args, **kwargs):
        super().__init__(in_planes, out_planes,*args, **kwargs)
        self.se = SEModule(out_planes * self.expansion)

    def forward(self, x):
        residual = x

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out = self.block(x)
        out = self.se(out )

        out.add_(residual)

        return out

class ResNetLayer(nn.Module):
    def __init__(self, in_planes, out_planes, depth, block=BasicBlock, *args, **kwargs):
        super().__init__()
        # if inputs == outputs then stride = 1, e.g 64==64 (first block)
        stride = 1 if in_planes == out_planes else 2

        expansion = block.expansion

        in_planes = in_planes * expansion if in_planes != out_planes else in_planes

        self.layer = nn.Sequential(
            block(in_planes, out_planes, stride=stride, *args, **kwargs),
            *[block(out_planes * block.expansion, out_planes, *args, **kwargs) for _ in range(max(0, depth - 1))],
        )

    def forward(self, x):
        out = self.layer(x)

        return out

class ResNet(nn.Module):
    def __init__(self, in_channel, blocks, block=BasicBlock, conv_layer=nn.Conv2d, *args, **kwargs):
        super().__init__()

        self.gate = nn.Sequential(
            conv_layer(in_channel, 64,  kernel_size=7, stride=2, padding=3,),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layers = nn.ModuleList([
            ResNetLayer(64, 64, depth=blocks[0], block=block, conv_layer=conv_layer, *args, **kwargs),
            ResNetLayer(64, 128, depth=blocks[1], block=block, conv_layer=conv_layer, *args, **kwargs),
            ResNetLayer(128, 256, depth=blocks[2], block=block, conv_layer=conv_layer, *args, **kwargs),
            ResNetLayer(256, 512, depth=blocks[3], block=block, conv_layer=conv_layer, *args, **kwargs),

        ])

        self.initialise(self.modules())

    @staticmethod
    def initialise(modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        trace = []

        x = self.gate(x)
        x = self.layers(x)

        return x


class TraversabilityResnet(nn.Module):
    def __init__(self, in_channel, blocks, block=BasicBlock, conv_layer=nn.Conv2d, *args, **kwargs):
        super().__init__()

        self.gate = nn.Sequential(
            conv_layer(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.encoder = nn.Sequential(
            ResNetLayer(64, 64, depth=blocks[0], block=block, conv_layer=conv_layer, *args, **kwargs),
            ResNetLayer(64, 128, depth=blocks[1], block=block, conv_layer=conv_layer, *args, **kwargs),
            ResNetLayer(128, 256, depth=blocks[2], block=block, conv_layer=conv_layer, *args, **kwargs),
            ResNetLayer(256, 512, depth=blocks[3], block=block, conv_layer=conv_layer, *args, **kwargs),
            nn.AvgPool2d(5)
        )

        self.decoder = nn.Sequential(
                                     nn.Linear(512 * block.expansion, 2))

        ResNet.initialise(self.modules())

    def forward(self, x):
        x = self.gate(x)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)

        return x
