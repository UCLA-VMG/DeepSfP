# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


class Timer(object):
    """Maintains and updates current time"""
    def __init__(self):
        self._time = None
        self.update()

    def update(self):
        self._time = time.time()

    @property
    def val(self):
        return self._time

    @property
    def toc(self):
        tic = self.val
        self.update()
        return self.val - tic


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def normalize_normals(normals):
    batch_size = len(normals)
    normals -= normals.view(batch_size,-1).min(dim=1)[0].view(-1,1,1,1)
    normals /= normals.view(batch_size,-1).max(dim=1)[0].view(-1,1,1,1)
    normals[normals != normals] = 0  # Replace NaN with 0
    return normals


def split_into_patches(input_img, roll_length, patch_size=256):
    """Split a test image into patches
    Args:
        input_img: 1 * C * H * H
        window_size: w
    Return:
        output_img: K * C * w * w (K = (H/w)**2）
    """
    # Roll image
    input_img = input_img.roll((roll_length,roll_length), (2, 3))

    # Create Patches
    img_patches = []
    block_num = input_img.shape[2] // patch_size
    for i in range(block_num):
        for j in range(block_num):
            x = i * patch_size
            y = j * patch_size
            img_patch = input_img[:, :, y:y + patch_size, x:x + patch_size]
            img_patches.append(img_patch)

    output_img = torch.cat(img_patches, 0)
    return output_img


def combine_patches(output_img, unroll_length):
    """Combine patches into the full image
    Args:
        output_img: K * C * w * w
    Return:
        full_img: 1 * C * H * H (H = s * sqrt(K)）
    """
    block_num = int(np.sqrt(output_img.shape[0]))
    patch_size = output_img.shape[2]

    full_img = torch.ones((1,
                           output_img.shape[1],
                           block_num * patch_size,
                           block_num * patch_size
                          ),device=output_img.device)

    for obj_idx in range(output_img.shape[0]):
        x = obj_idx // block_num * patch_size
        y = obj_idx % block_num * patch_size
        full_img[:, :, y:y + patch_size, x:x + patch_size] = output_img[obj_idx, :, :, :]

    return full_img.roll((-unroll_length,-unroll_length), (2, 3))  # Unroll image
