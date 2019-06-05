import torch
import torch.nn as nn

from torchvision import models
import torch.nn.functional as F


class DoNothing(nn.Module):
    def forward(self, x):
        return x


def conv_block3x3(in_planes, out_planes, conv_layer=nn.Conv2d,
                  kernel_size=3, padding=None, stride=1):
    padding = kernel_size // 2 if not padding else padding

    return conv_layer(in_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False, stride=stride)

    #
    # if prectivate:
    #     conv_block = nn.Sequential(
    #         nn.BatchNorm2d(in_planes),
    #         activation_funcs[activation],
    #         conv_layer(in_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False, stride=stride)
    #     )
    # else:
    #     conv_block = nn.Sequential(
    #         conv_layer(in_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False, stride=stride),
    #         nn.BatchNorm2d(out_planes),
    #         activation_funcs[activation],
    #     )
    #
    # return conv_block


class SEModule(nn.Module):
    """
    Squeeze and Excitation module https://arxiv.org/abs/1709.01507
    """

    def __init__(self, n_features, ratio=8, *args, **kwargs):
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


def get_activation_funcs():
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)]
    ])


class BasicBlock(nn.Module):
    """
    Basic block of ResNet, it is composed by two 3x3 conv - relu and batchnorm.
    It automatically performs residual addition
    """
    expansion = 1


    def __init__(self, in_planes, out_planes, stride=1, conv_block=conv_block3x3, activation='relu', preactivate=False,
                 *args, **kwargs):
        super().__init__()
        self.in_planes, self.out_planes, self.conv_block, self.stride = in_planes, out_planes, conv_block, stride

        self.shortcut = self.get_shortcut() if self.in_planes != self.expanded else None


        if preactivate:
            self.convs = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                self.activations_funcs[activation],
                conv_block(in_planes, out_planes, stride=stride),
                nn.BatchNorm2d(out_planes),
                self.activations_funcs[activation],
                conv_block(out_planes, out_planes),
            )

        else:
            self.convs = nn.Sequential(conv_block(in_planes, out_planes, stride=stride),
                                       nn.BatchNorm2d(out_planes),
                                       self.activations_funcs[activation],
                                       conv_block(out_planes, out_planes),
                                       nn.BatchNorm2d(out_planes))

        self.preactivate, self.activation = preactivate, activation
    @property
    def activations_funcs(self):
        return nn.ModuleDict([
            ['relu', nn.ReLU(inplace=True)],
            ['leaky_relu', nn.LeakyReLU(negative_slope=0.1, inplace=True)],
            ['selu', nn.SELU(inplace=True)]
        ])

    @property
    def expanded(self):
        return self.out_planes * self.expansion

    def get_shortcut(self):
        return nn.Sequential(
            nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1,
                      stride=self.stride, bias=False),
            nn.BatchNorm2d(self.out_planes),
        )

    def forward(self, x):
        residual = x
        if self.shortcut is not None: residual = self.shortcut(residual)
        out = self.convs(x)
        out += residual
        if not self.preactivate:
            out = self.activations_funcs[self.activation](out)

        return out


class Bottleneck(BasicBlock):
    expansion = 4

    def get_convs(self, in_planes, out_planes, stride, conv_block=conv_block3x3, *args, **kwargs):
        return nn.Sequential(
            conv_block3x3(in_planes, out_planes, kernel_size=1),
            conv_block3x3(out_planes, out_planes, kernel_size=3, stride=stride),
            conv_block3x3(out_planes, self.expanded, kernel_size=1),
        )


class BasicBlockSE(BasicBlock):
    def __init__(self, in_planes, out_planes, ratio=16, *args, **kwargs):
        super().__init__(in_planes, out_planes, *args, **kwargs)
        self.se = SEModule(out_planes, ratio)

    def forward(self, x):
        residual = x
        if self.shortcut is not None: residual = self.shortcut(residual)
        out = self.convs(x)
        out = self.se(out)
        out += residual
        if not self.preactivate:
            out = self.activations_funcs[self.activation](out)

        return out


class BottleneckSE(Bottleneck):
    def __init__(self, in_planes, out_planes, ratio=16, *args, **kwargs):
        super().__init__(in_planes, out_planes, *args, **kwargs)
        self.se = SEModule(out_planes * self.expansion, ratio)

    def forward(self, x):

        residual = x
        if self.shortcut is not None: residual = self.shortcut(residual)
        out = self.convs(x)
        out = self.se(out)
        out += residual
        if not self.preactivate:
            out = self.activations_funcs[self.activation](out)

        return out


class ResNetLayer(nn.Module):
    """
    This class represent a layer of ResNet, it stacks a number of blocks
    equal to the depth parameter. If the `in_planes` and `out_planes` are different,
    for example 64 features in and 128 features out, the first layer will use a stride of two,
    quoting from the paper: 'We perform downsampling directly by convolutional layers that have a stride of 2.'
    """

    def __init__(self, in_planes, out_planes, depth, block=BasicBlock, *args, **kwargs):
        super().__init__()
        # if inputs == outputs then stride = 1, e.g 64==64 (first block)
        should_downsample = in_planes != out_planes
        stride = 2 if should_downsample else 1

        # create the layer by directly instantiate the first block with the correct stride and then
        # if needed create all the others blocks
        self.layer = nn.Sequential(
            block(in_planes * block.expansion, out_planes, stride=stride, *args, **kwargs),
            *[block(out_planes * block.expansion, out_planes, *args, **kwargs) for _ in range(max(0, depth - 1))],
        )

    def forward(self, x):
        out = self.layer(x)

        return out


class ResNetEncoder(nn.Module):
    """
    This class represents the head of ResNet. It reduce the dimension of the input image by apply the
    .gate layer first and feed the output to one layer after the other.
    """

    def __init__(self, in_channel, depths, blocks=BasicBlock, blocks_sizes=None, conv_block=conv_block3x3,
                 activation='relu', *args,
                 **kwargs):
        super().__init__()
        self.in_channel, self.depths, self.blocks = in_channel, depths, blocks

        self.blocks_sizes = blocks_sizes if blocks_sizes is not None else [(64, 64), (64, 128), (128, 256), (256, 512)]

        self.gate = nn.Sequential(
            nn.Conv2d(in_channel, self.blocks_sizes[0][0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0][0]),
            get_activation_funcs()[activation],
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # if the user passed a single instance of block, use it for each layer
        if type(blocks) is not list:
            self.blocks = [blocks] * len(self.blocks_sizes)
        # stack a number of layers together equal to the len of depth
        self.layers = nn.ModuleList([
            ResNetLayer(in_c, out_c, depth=depths[i], block=self.blocks[i], activation=activation, *args, **kwargs)
            for i, (in_c, out_c) in enumerate(self.blocks_sizes)
        ])

        self.initialise(self.modules(), activation)

    @staticmethod
    def initialise(modules, activation='relu'):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.gate(x)

        for layer in self.layers:
            x = layer(x)

        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """

    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):
    """
    ResNet https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, depths, in_channel=3, blocks=BasicBlock, n_classes=1000, encoder=ResNetEncoder,
                 decoder=ResnetDecoder, *args, **kwargs):
        super().__init__()
        self.depths, self.in_channel, self.n_classes = depths, in_channel, n_classes
        self.encoder = encoder(in_channel, depths, blocks, *args, **kwargs)
        self.decoder = decoder(self.encoder.blocks_sizes[-1][1] * self.encoder.blocks[-1].expansion, n_classes)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def resnet18(in_channel=3, blocks=BasicBlock, resnet=ResNet, pretrained=False, *args, **kwargs):
    model = resnet([2, 2, 2, 2], in_channel, blocks, *args, **kwargs)

    if pretrained:
        print('loading trained weights...')
        restore(models.resnet18(True), model)

    return model


def resnet34(in_channel=3, blocks=BasicBlock, pretrained=False, resnet=ResNet, **kwargs):
    model = resnet([3, 4, 6, 3], in_channel, blocks, **kwargs)

    if pretrained:
        print('loading trained weights...')
        restore(models.resnet34(True), model)

    return model


def resnet50(in_channel=3, blocks=Bottleneck, pretrained=False, resnet=ResNet, **kwargs):
    model = resnet([3, 4, 6, 3], in_channel, blocks, **kwargs)
    if pretrained:
        print('loading trained weights...')
        restore(models.resnet50(True), model)

    return model


def resnet101(in_channel=3, blocks=Bottleneck, pretrained=False, resnet=ResNet, **kwargs):
    model = resnet([3, 4, 23, 3], in_channel, blocks, **kwargs)
    if pretrained:
        print('loading trained weights...')
        restore(models.resnet101(True), model)
    return model


def resnet152(in_channel=3, blocks=Bottleneck, pretrained=False, resnet=ResNet, **kwargs):
    model = resnet([3, 8, 36, 3], in_channel, blocks, **kwargs)
    if pretrained:
        print('loading trained weights...')
        restore(models.resnet152(True), model)
    return model


def restore(source, target):
    pre_trained_layers = [source.layer1, source.layer2, source.layer3, source.layer4]

    for i, pre_trained_layer in enumerate(pre_trained_layers):
        layer = target.encoder.layers[i]
        p_t_convs = [m for m in pre_trained_layer.modules() if isinstance(m, nn.Conv2d)]
        p_t_bns = [m for m in pre_trained_layer.modules() if isinstance(m, nn.BatchNorm2d)]

        convs = [m for m in layer.modules() if isinstance(m, nn.Conv2d)]
        bns = [m for m in layer.modules() if isinstance(m, nn.BatchNorm2d)]

        for p_t_conv, conv in zip(p_t_convs, convs):
            conv.load_state_dict(p_t_conv.state_dict())

        for p_t_bn, bn in zip(p_t_bns, bns):
            bn.load_state_dict(p_t_bn.state_dict())
