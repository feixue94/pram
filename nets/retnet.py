# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> retnet
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   22/02/2024 15:23
=================================================='''
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   glretrieve -> retnet
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   15/02/2024 10:55
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, groups=32, dilation=1, norm_layer=None, ac_fn=None):
        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, outplanes)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = conv3x3(outplanes, outplanes, stride, groups, dilation)
        self.bn2 = norm_layer(outplanes)
        self.conv3 = conv1x1(outplanes, outplanes)
        self.bn3 = norm_layer(outplanes)
        if ac_fn is None:
            self.ac_fn = nn.ReLU(inplace=True)
        else:
            self.ac_fn = ac_fn

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ac_fn(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ac_fn(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.ac_fn(out)

        return out


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.p) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class RetNet(nn.Module):
    def __init__(self, indim=256, outdim=1024):
        super().__init__()

        ac_fn = nn.GELU()

        self.convs = nn.Sequential(
            # no batch normalization

            nn.Conv2d(in_channels=indim, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            # nn.ReLU(),

            ResBlock(512, 512, groups=32, stride=1, ac_fn=ac_fn),
            ResBlock(512, 512, groups=32, stride=1, ac_fn=ac_fn),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            # nn.ReLU(),
            ResBlock(inplanes=1024, outplanes=1024, groups=32, stride=1, ac_fn=ac_fn),
            ResBlock(inplanes=1024, outplanes=1024, groups=32, stride=1, ac_fn=ac_fn),
        )

        self.pool = GeneralizedMeanPoolingP()
        self.fc = nn.Linear(1024, out_features=outdim)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.convs(x)
        out = self.pool(out).reshape(x.shape[0], -1)
        out = self.fc(out)
        out = F.normalize(out, p=2, dim=1)
        return out


if __name__ == '__main__':
    mode = RetNet(indim=256, outdim=1024)
    state_dict = mode.state_dict()
    keys = state_dict.keys()
    print(keys)
    shapes = [state_dict[v].shape for v in keys]
    print(shapes)
