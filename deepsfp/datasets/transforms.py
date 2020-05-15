# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
import torch
import numpy as np
import os
from typing import Union, Tuple

from .factory import transforms_factory

__all__ = ['RandomCrop']


@transforms_factory.register()
class RandomCrop(object):
    """
    Apply a (pre-generated) random crop of given shape and minimum forground ratio 
    to the image in a sample. Read more on generating random crop idices 
    [here](https://github.com/alexrgilbert/deepsfp/blob/master/README.md). 

    :param dataset_dir (Union[os.PathLike,str]): Parent directory of the dataset (see DATASET.DIR and DATASET.DATASET config fields)
    :param output_size (Union[tuple,int]): Desired crop shape. Accepts either (height, width) or side length of a square crop.
    :param object_ratio_threshold (float): Minimum foreground (i.e. object) ratio of valid crops
    """
    def __init__(self, dataset_dir: Union[os.PathLike,str], crop_size: Union[Tuple,int] = (256, 256), 
                    foreground_ratio_threshold: float = 0.4):
        self._crop_h, self._crop_w = crop_size
        self._thresh = foreground_ratio_threshold
        self._crop_mask_dir = os.path.join(dataset_dir,'crop_indices',f'{self._crop_h}_{self._crop_w}_{self._thresh}')
        if not os.path.isdir(self._crop_mask_dir):
            raise ValueError(f'Invalid crop mask directory! {self._crop_mask_dir}')

    def __call__(self, item):
        idx, obj_filename, obj_name, sample = item
        crop_pth = os.path.join(self._crop_mask_dir,f'{obj_filename}.pth')
        crop_idcs = torch.load(crop_pth)
        idx = np.random.randint(crop_idcs.shape[0])
        top,left = crop_idcs[idx,:].data
        for key, value in sample.items():
            sample[key] = value[:,top:top + self._crop_h,left:left + self._crop_w]
        return (idx, obj_filename, obj_name, sample)
