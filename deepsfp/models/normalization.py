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
import torch
import torch.nn as nn
import torch.nn.functional as F

from .factory import norm_factory, actvn_factory

__all__ = ['SPADE']


@norm_factory.register()
class SPADE(nn.Module):
    """
    Spatially-Adaptive Normalization block.
      Based on https://arxiv.org/abs/1903.07291
    """
    def __init__(self, inplanes: int, mlp_inplanes: int = 4, nhidden: int = 128, 
                kernel_size: int = 3, padding: int = 1, interp_mode: str = 'bilinear', 
                activation: Union[Dict, CfgNode] = {}, normalization: Union[Dict, CfgNode] = {}):
        super().__init__()

        self.interp = partial(F.interpolate, mode=interp_mode)
        self.mlp_conv = nn.Conv2d(mlp_inplanes, nhidden, kernel_size, padding=padding,)
        self.mlp_act = actvn_factory.build(**activation)  # nn.ReLU(inplace=True)
        self.mlp_gamma = nn.Conv2d(nhidden, inplanes, kernel_size, padding=padding,)
        self.mlp_beta = nn.Conv2d(nhidden, inplanes, kernel_size, padding=padding,)
        self.norm = norm_factory.build(**normalization)(inplanes)  # nn.BatchNorm2d(inplanes, affine=False)

    def forward(self, x, polar):
        # Produce scaling and bias conditioned on polar images
        polar = self.interp(polar, size = x.shape[2:])
        polar = self.mlp_conv(polar)
        polar = self.mlp_act(polar)
        gamma = self.mlp_gamma(polar)
        beta = self.mlp_beta(polar)

        out = self.norm(x)  # generate parameter-free normalized activations
        out = (1 + gamma) * out + beta  # apply conditioned scale & bias

        return out