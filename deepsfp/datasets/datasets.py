# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
import os
import pandas as pd
from typing import Union
import torch
import torch.utils.data as data
from torchvision.transforms import Compose

from .factory import dataset_factory

__all__ = ['SurfaceNormals']


@dataset_factory.register()
class SurfaceNormals(data.Dataset):
    def __init__(self, root: Union[str, os.PathLike], data_list: Union[str, os.PathLike],
                  transform: Compose = Compose([]), **kwargs):
        super().__init__()
        
        datadir = self.__class__.__name__
        datafile = os.path.join(root, datadir, data_list)
        self._data_list = pd.read_csv(datafile, header=None, squeeze=True)
        self._obj_dir = os.path.join(root, datadir, 'objects')

        self._transform = transform

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        obj_filename = os.path.splitext(self._data_list[idx])[0]
        sample = torch.load(os.path.join(self._obj_dir, f'{obj_filename}.pth'))
        obj_name = obj_filename.replace('/','_')
        return self._transform((idx, obj_filename, obj_name, sample))
