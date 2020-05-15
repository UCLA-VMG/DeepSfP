# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
import argparse
import os
from typing import Union
from tqdm import tqdm
from glob import glob
import logging
import numpy as np
import scipy.io as sio
import torch
import pandas as pd
from __init__ import TqdmLoggingHandler
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deepsfp.config import config, update_config
from deepsfp.utils import SfParser, setup_experiment


def parse_args():
    parser = argparse.ArgumentParser(description='Tool for converting from Matlab \
                                                files of published dataset to format \
                                                expected by dataloader', parents=[SfParser(add_help=False)])
    parser.add_argument('--overwrite', '-o', help='Whether to overwrite existing converted \
                         files', action='store_true')
    parser.add_argument('--debug', '-d', help='Run without saving', action='store_true')
    args = parser.parse_args()
    return args

class DataConverter:
    def __init__(self, root: Union[str, os.PathLike], name: str, data_list: Union[str, os.PathLike], 
                    precision: Union[str, torch.dtype] = torch.float32, **kwargs):
        self._data_dir = os.path.join(root, name)
        self._data_list = data_list
        self._precision = precision

    def _toTensor(self, arr, mask, precision = None):
        '''
        Convert to Cx1024x1024 tensors consumable by model
        '''
        arr = np.atleast_3d(arr)  # expand to 3d
        arr = arr * np.atleast_3d(mask)  # apply binary mask
        arr = arr[:,100:1124,:]  # remove padding
        arr = arr.transpose((2,0,1))  # transpose to CxHxW
        if not precision:
            precision = self._precision
        arr = arr.astype(self._precision)  # convert precision
        arr = torch.from_numpy(arr)  # convert to tensor
        return arr

    def __repr__(self):
        dataset = f'Dataset @ {self._data_dir}'
        objects = f'Objects in {self._data_list}'
        return f'DataConverter:\n\t{dataset}\n\t{objects}'

    def run(self, logger: logging.Logger, overwrite: bool = False, debug: bool = False):
        logger.info(f'Running {self}...')
        obj_dir = os.path.join(self._data_dir, 'objects')
        mat_list = pd.read_csv(os.path.join(self._data_dir, self._data_list), header=None, squeeze=True)
        pth_list = mat_list.apply(lambda f: f'{os.path.splitext(f)[0]}.pth')
        filenames = zip(mat_list, pth_list)
        if not overwrite:
            filenames = filter(lambda mp: not os.path.exists(os.path.join(obj_dir,mp[-1])),filenames)
        pbar = tqdm(list(filenames))
        for i,(mat_path,pth_path) in enumerate(pbar):
            pbar.set_description(f'{mat_path} -> {pth_path}')
            logger.info(f'[{i+1}/{len(pth_list)}] Converting {mat_path} -> {pth_path}...')
            mat_file = sio.loadmat(os.path.join(obj_dir,mat_path))
            mask = mat_file['mask']
            sample = {
                'image': self._toTensor(mat_file['images'], mask),
                'est': self._toTensor(mat_file['normals_prior'], mask),
                'label': self._toTensor(mat_file['normals_gt'], mask),
                'mask': self._toTensor(np.stack((mask,mask,mask),axis=2), mask, 'uint8'),
            }
            if not debug:
                dest = os.path.join(obj_dir,pth_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                torch.save(sample, dest)
                logger.info(f'\tSaving to {os.path.realpath(dest)}')


def setup_by_phase(config, args, phase='train'):
    logger, exp_dir, _, _, _ = setup_experiment(config, args.config, phase=phase, 
                                    name=f'{phase}-set-mat-to-pth', meta=[args, config])
    logger.removeHandler(logger.handlers[-1])
    logger.addHandler(TqdmLoggingHandler())
    logger.info(f'See {exp_dir} for {phase} set logs...')
    return logger


def main():
    '''
    Load mat files, convert fields, and save torch tensor to .pth file
    '''
    args = parse_args()
    update_config(config, config_filepath=args.config, cli_options=(args.opts + ['enable_tblogging', 'False']))

    # Train set
    logger = setup_by_phase(config, args, 'train')
    trainset_cig = DataConverter(**config.train.dataloader.dataset)
    trainset_cig.run(logger, args.overwrite, args.debug)

    # Test set
    logger = setup_by_phase(config, args, 'test')
    testset_cig = DataConverter(**config.test.dataloader.dataset)
    testset_cig.run(logger, args.overwrite, args.debug)


if __name__ == '__main__':
    main()