# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
from torch import nn
import torch.optim
import torch.optim.lr_scheduler
from torch.optim.optimizer import Optimizer

from ..utils import ComponentFactory


class OptimizerFactory(ComponentFactory):
    def __init__(self):
        super().__init__('optimizer', modules=[torch.optim], default='SGD')

    def build(self, model: nn.Module, name: str = '', lr: float = 1e-2, **optimizer_cfg):
        params = filter(lambda p: p.requires_grad, model.parameters())
        return super().build(name, params=params, lr=lr, **optimizer_cfg)


class LRSchedulerFactory(ComponentFactory):
    def __init__(self):
        super().__init__('learning rate scheduler', modules=[torch.optim.lr_scheduler], allow_none=True)

    def build(self, optimizer: Optimizer, name: str = '', **lr_scheduler_cfg):
        return super().build(name, optimizer=optimizer, **lr_scheduler_cfg)


optimizer_factory = OptimizerFactory()
lr_scheduler_factory = LRSchedulerFactory()
loss_factory = ComponentFactory('loss', modules=[nn])
metric_factory = ComponentFactory('error metric', modules=[nn])
