# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pformat
from typing import Dict
import torch.nn
import torch.nn.init

from ..utils import ComponentFactory


initializer_factory = ComponentFactory('model weights initializer', partial=True, modules=[torch.nn.init], default='xavier_normal_')
block_factory = ComponentFactory('model block', partial=True, modules=[torch.nn])

class ModelFactory(ComponentFactory):
    def build(self, name: str = '', logger=None, initializer: Dict = {}, blocks: Dict = {}, **model_cfg):
        initializer = initializer_factory.build(**initializer)
        blocks = {lvl: block_factory.build(**cfg) for lvl,cfg in blocks.items()}
        model = super().build(name, initializer=initializer, **blocks, **model_cfg)
        model.init_weights(logger)
        logger.info(pformat(model))
        return model

model_factory = ModelFactory()
actvn_factory = ComponentFactory('activation function', modules=[torch.nn], default='ReLU')
norm_factory = ComponentFactory('normalization function', partial=True, modules=[torch.nn], default='BatchNorm2d')
