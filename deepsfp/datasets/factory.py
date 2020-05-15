# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose

from ..utils import ComponentFactory


class TransformsFactory(ComponentFactory):
    def __init__(self):
        super().__init__('transform', modules=[transforms])

    def build(self, **transforms_cfg):
        _build = lambda kv: super(TransformsFactory, self).build(kv[0], **kv[1])
        return Compose(list(map(_build,transforms_cfg.items())))

transforms_factory = TransformsFactory()


class DatasetFactory(ComponentFactory):
    def __init__(self):
        super().__init__('dataset')

    def build(self, transforms = {}, **dataset_cfg):
        transform = transforms_factory.build(**transforms)
        return super().build(transform=transform, **dataset_cfg)

dataset_factory = DatasetFactory()


class DataloaderFactory(ComponentFactory):
    def __init__(self):
        super().__init__('dataloader')

    def build(self, dataset, **dataloader_cfg):
        dataset = dataset_factory.build(**dataset)
        return DataLoader(dataset, **dataloader_cfg)

dataloader_factory = DataloaderFactory()
