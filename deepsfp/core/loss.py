# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn

from .factory import loss_factory

__all__ = ['CosineEmbeddingLoss']


@loss_factory.register()
class CosineEmbeddingLoss(nn.Module):
    def __init__(self,mask_flag: bool = True, reduction: str = 'sum'):
        super(CosineEmbeddingLoss, self).__init__()
        self.criterion = nn.CosineEmbeddingLoss(reduction=reduction)
        self._masked = mask_flag

    def __call__(self, output, target, mask):
        output = output.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        target = target.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        if self._masked:
            mask = mask.float().permute(0, 2, 3, 1).contiguous().view(-1, 3)
            self.loss = self.criterion((output*mask), (target*mask), (2 * mask[:,1]) - 1)
            num_val = torch.sum(mask[:,1])
            self.loss = (self.loss / num_val) if num_val else (self.loss * num_val)
        else:
            flag = torch.ones_like(output[:,0])
            self.loss = self.criterion(output, target, flag)
            self.loss /= flag.numel()
        return self.loss

    def backward(self):
        self.loss.backward()
