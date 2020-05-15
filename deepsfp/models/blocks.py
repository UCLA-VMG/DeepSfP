# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode
from functools import partial
from typing import Union, Dict
import torch.nn as nn
import torch.nn.functional as F

from .factory import block_factory, norm_factory, actvn_factory

__all__ = ['SfPEncoderBlock', 'SfPDecoderBlock', 'SfPHeadBlock']


@block_factory.register()
class SfPEncoderBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, kernel_size: int = 3, padding: int = 1, 
                activation: Union[Dict, CfgNode] = {}, normalization: Union[Dict, CfgNode] = {}):
        super().__init__()

        self._norm = norm_factory.build(**normalization)
        self.act = actvn_factory.build(**activation)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size, padding=padding, stride=2,)
        self.norm1 = self._norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size, padding=padding,)
        self.norm2 = self._norm(planes)

        self.downsample = nn.Conv2d(inplanes, planes, 1, stride=2)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out += identity
        out = self.act(out)

        return out


@block_factory.register()
class SfPDecoderBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, kernel_size: int = 3, padding: int = 1, 
                interp_mode: str = 'bilinear', activation: Union[Dict, CfgNode] = {}, 
                normalization: Union[Dict, CfgNode] = {}):
        super().__init__()

        self._norm = norm_factory.build(**normalization)
        self.act = actvn_factory.build(**activation)

        self.upsample = partial(F.interpolate, scale_factor=2, mode=interp_mode)
        self.bottleneck = nn.Conv2d(inplanes, planes, 1)

        self.norm1 = self._norm(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size, padding=padding,)
        self.norm2 = self._norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size, padding=padding)

    def forward(self, x, polar):
        x = self.upsample(x)

        identity = self.bottleneck(x)

        out = self.norm1(x, polar)
        out = self.act(out)
        out = self.conv1(out)
        out = self.norm2(out, polar)
        out = self.act(out)
        out = self.conv2(out)

        out += identity

        return out
        

@block_factory.register()
class SfPHeadBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, kernel_size: int = 3, padding: int = 1, 
                 activation: Union[Dict, CfgNode] = {}, normalization: Union[Dict, CfgNode] = {}):
        super().__init__()

        self._norm = norm_factory.build(**normalization)
        self.act = actvn_factory.build(**activation)

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, padding=padding)
        self.norm1 = self._norm(inplanes)
        
        self.conv_out = nn.Conv2d(inplanes, planes, kernel_size, padding=padding,)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv_out(out + identity)
        out = F.normalize(out)

        return out
