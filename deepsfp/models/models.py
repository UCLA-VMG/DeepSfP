# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from typing import Callable
import torch
import torch.nn as nn

from .factory import model_factory

__all__ = ['SfPNet']


@model_factory.register()
class SfPNet(nn.Module):
    def __init__(self, initializer: "partial[Callable]", encoder: "partial[nn.Module]", 
                    decoder: "partial[nn.Module]", head: "partial[nn.Module]"):
        super().__init__()

        self._initializer = initializer

        self.enc1 = encoder(13, 32)
        self.enc2 = encoder(32, 64)
        self.enc3 = encoder(64, 128)
        self.enc4 = encoder(128, 256)
        self.enc5 = encoder(256, 512)

        self.dec1 = decoder(512, 256)
        self.dec2 = decoder(256, 128)
        self.dec3 = decoder(128, 64)
        self.dec4 = decoder(64, 32)
        self.dec5 = decoder(32, 24)

        self.head = head(24, 3)

    def forward(self, polar, prior):
        inp = torch.cat((prior, polar), dim=1)

        x1 = self.enc1(inp)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        out = self.dec1(x5 + x5, polar)
        out = self.dec2(out + x4, polar)
        out = self.dec3(out + x3, polar)
        out = self.dec4(out + x2, polar)
        out = self.dec5(out + x1, polar)

        out = self.head(out)
        return out

    def init_weights(self,logger=None):
        if logger is not None:
            logger.info('=> initializing model weights...')
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                self._initializer(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
