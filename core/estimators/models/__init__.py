from .custom_resnet import *

zoo = {
    'microresnet#4-gate=3x3-n=2-se=True': MicroResnet.micro(1,
                                                            n=2,
                                                            encoder=Encoder3x3,
                                                            blocks=[BasicBlock, BasicBlock,
                                                                    BasicBlockSE],
                                                            preactivate=True,
                                                            activation='leaky_relu'),
    'microresnet#4-gate=3x3-n=1-se=True': MicroResnet.micro(1,
                                                            n=1,
                                                            encoder=Encoder3x3,
                                                            blocks=[BasicBlock, BasicBlock,
                                                                    BasicBlockSE],
                                                            preactivate=True,
                                                            activation='leaky_relu'),
    'microresnet#4-gate=3x3-n=5-se=True': MicroResnet.micro(1,
                                                            n=5,
                                                            encoder=Encoder3x3,
                                                            blocks=[BasicBlock, BasicBlock, BasicBlock,
                                                                    BasicBlockSE],
                                                            preactivate=True,
                                                            activation='leaky_relu'),
    'microresnet#4-gate=7x7-n=2-se=True': MicroResnet.micro(1,
                                                            n=2,
                                                            encoder=Encoder7x7,
                                                            blocks=[BasicBlock, BasicBlock, BasicBlock,
                                                                    BasicBlockSE],
                                                            preactivate=True,
                                                            activation='leaky_relu'),
    'microresnet#4-gate=7x7-n=1-se=True': MicroResnet.micro(1,
                                                            n=1,
                                                            encoder=Encoder7x7,
                                                            blocks=[BasicBlock, BasicBlock, BasicBlock,
                                                                    BasicBlockSE],
                                                            preactivate=True,
                                                            activation='leaky_relu'),
    'microresnet#4-gate=5x5-n=2-se=True': MicroResnet.micro(1,
                                                            n=2,
                                                            encoder=Encoder5x5,
                                                            blocks=[BasicBlock, BasicBlock, BasicBlock,
                                                                    BasicBlockSE],
                                                            preactivate=True,
                                                            activation='leaky_relu'),

    'micro2resnet#4-gate=3x3-n=2-se=True': MicroResnet.micro2(1,
                                                            n=2,
                                                            encoder=Encoder3x3,
                                                            blocks=[BasicBlock, BasicBlock, BasicBlock,
                                                                    BasicBlockSE],
                                                            preactivate=True,
                                                            activation='leaky_relu')



}



