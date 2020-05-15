# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn

from .factory import metric_factory

__all__ = ['MeanAngularError']


@metric_factory.register()
class MeanAngularError(nn.Module):
    def __init__(self, mask_flag):
        super(MeanAngularError, self).__init__()
        self._masked = mask_flag

    @torch.no_grad()
    def __call__(self, output, target, mask):
        dot_product = (output * target).sum(1)
        output_norm = torch.norm(output, 2, 1)
        target_norm = torch.norm(target, 2, 1)
        dot_product = (dot_product / (output_norm * target_norm + 1e-8)).clamp(-1, 1)
        angular_map = torch.acos(dot_product) * (180.0 / math.pi)  # [-180, 180]
        
        if self._masked:
            mask = mask.narrow(1, 0, 1).squeeze(1)
            total_ae = angular_map[mask.byte()].sum()
            tot = mask.sum()
            return (total_ae / tot) if tot else torch.zeros_like(total_ae)
        else:
            return torch.mean(angular_map)