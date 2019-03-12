from .custom_resnet import *


zoo = {'microresnet#4-gate=5x5-n=2-se=True': MicroResnet.micro(1,
                                                                  n=2,
                                                                  blocks=[BasicBlock, BasicBlock, BasicBlock,
                                                                          BasicBlockSE],
                                                                  preactivate=True,
                                                                  activation='leaky_relu')}
