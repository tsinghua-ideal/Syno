'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet8  |    8  | 0.075M
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4M

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import os
import sys

from KAS import Placeholder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.models import mapping_func_conv, KASModule

__all__ = ['KASResNet', 'resnet8', 'resnet20', 'resnet32',
           'resnet44', 'resnet56', 'resnet110', 'resnet1202']


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class KASBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(KASBasicBlock, self).__init__()
        self.conv1 = Placeholder(
            refered_layer=nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            mapping_func=mapping_func_conv
        ) if stride == 1 else nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Placeholder(
            refered_layer=nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            mapping_func=mapping_func_conv
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    Placeholder(
                        refered_layer=nn.Conv2d(in_planes, self.expansion * planes,
                                                kernel_size=1, stride=stride, bias=False),
                        mapping_func=mapping_func_conv
                    ),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class KASResNet(KASModule):
    def __init__(self, block, num_blocks, num_classes=10):
        super(KASResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = Placeholder(
            refered_layer=nn.Conv2d(3, 16, kernel_size=3,
                                    stride=1, padding=1, bias=False),
            mapping_func=mapping_func_conv
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet8():
    return KASResNet(KASBasicBlock, [1, 1, 1])


def resnet20():
    return KASResNet(KASBasicBlock, [3, 3, 3])


def resnet32():
    return KASResNet(KASBasicBlock, [5, 5, 5])


def resnet44():
    return KASResNet(KASBasicBlock, [7, 7, 7])


def resnet56():
    return KASResNet(KASBasicBlock, [9, 9, 9])


def resnet110():
    return KASResNet(KASBasicBlock, [18, 18, 18])


def resnet1202():
    return KASResNet(KASBasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(
        lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
