# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
import argparse
import os
from typing import Union, Dict
from tqdm import tqdm
from glob import glob
import scipy.io as sio
import torch
import pandas as pd
from yacs.config import CfgNode
import logging
from __init__ import TqdmLoggingHandler
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deepsfp.config import config, update_config
from deepsfp.utils import SfParser, setup_experiment


def parse_args():
    parser = argparse.ArgumentParser(description='Tool for generating crop indices for each \
                                                object in a dataset', parents=[SfParser(add_help=False)])
    parser.add_argument('--overwrite', '-o', help='Whether to overwrite existing indices for this \
                        dataset/dimension/threshold combination.', action='store_true')
    parser.add_argument('--debug', '-d', help='Run without saving.', action='store_true')
    args = parser.parse_args()
    return args


class CropIndexGenerator:
    def __init__(self, data_list: Union[str, os.PathLike], transforms: Union[Dict, CfgNode], **kwargs):
        '''Generate crop-indices for all valid crops (according to configured crop
        dimensions/foreground-threshold) for each item in configured dataset.
        '''
        self._data_list = data_list
        self._crop_cfg = transforms.get('RandomCrop')
        if self._crop_cfg:
            self._crop_h, self._crop_w = self._crop_cfg.crop_size
            self._thresh = self._crop_cfg.foreground_ratio_threshold
            self._data_dir = self._crop_cfg.dataset_dir

            self._sum_filter = torch.nn.Conv2d(1,1, (self._crop_h, self._crop_w))
            for m in self._sum_filter.parameters():
                m.requires_grad = False
            self._sum_filter.weight.data.fill_(1)
            self._sum_filter.bias.data.zero_()
            self._sum_filter.cuda()

    @property
    def no_op(self):
        return self._crop_cfg is None

    @property
    def crop_w(self):
        if self.no_op:
            return None
        return self._crop_w

    @property
    def crop_h(self):
        if self.no_op:
            return None
        return self._crop_h

    @property
    def thresh(self):
        if self.no_op:
            return None
        return self._thresh

    @property
    def pixel_thresh(self):
        if self.no_op:
            return None
        return self.thresh * self.crop_h * self.crop_w

    @property
    def data_dir(self):
        if self.no_op:
            return None
        return self._data_dir

    @property
    def id(self):
        return f'{self.crop_h}_{self.crop_w}_{self.thresh}'
    
    def __repr__(self,):
        bounds = f'Bounds = {self.crop_h}x{self.crop_w}'
        thresh = f'Foreground thresh = {self.thresh}'
        dataset = f'Dataset @ {self.data_dir}'
        objects = f'Objects in {self._data_list}'
        return f'CropIndexGenerator:\n\t{bounds}\n\t{thresh}\n\t{dataset}\n\t{objects}'

    @torch.no_grad()
    def _generate_idcs(self, sample):
        mask = sample['mask'][0:1].float().unsqueeze(0).cuda()
        crop_mask = self._sum_filter(mask).squeeze() > self.pixel_thresh
        crop_idcs = torch.nonzero(crop_mask).cpu()
        return crop_idcs

    def run(self, logger: logging.Logger, overwrite: bool = False, debug: bool = False):
        if self.no_op:
            logger.info(f'random_crop not found in transforms config! Skipping!')
            return
        logger.info(f'Running {self}...')
        obj_dir = os.path.join(self.data_dir, 'objects')
        idcs_dir = os.path.join(self.data_dir, 'crop_indices', self.id)
        mat_list = pd.read_csv(os.path.join(self.data_dir, self._data_list), header=None, squeeze=True)
        pth_list = mat_list.apply(lambda f: f'{os.path.splitext(f)[0]}.pth')
        if not overwrite:
            pth_list = filter(lambda f: not os.path.exists(os.path.join(idcs_dir,f)), pth_list)
        pbar = tqdm(list(pth_list))
        for i,file_name in enumerate(pbar):
            pbar.set_description(f'{file_name}')
            logger.info(f'[{i+1}/{len(pth_list)}] Generating idcs for {file_name}...')
            sample = torch.load(os.path.join(obj_dir,file_name))
            crop_idcs = self._generate_idcs(sample)
            if not debug:
                dest = os.path.join(idcs_dir,file_name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                torch.save(crop_idcs, dest)
                logger.info(f'\tSaving to {os.path.realpath(dest)}')


def setup_by_phase(config, args, phase='train'):
    logger, exp_dir, _, _, _ = setup_experiment(config, args.config, phase=phase, 
                                    name=f'{phase}-set-crop-mask-generation', meta=[args, config])
    logger.removeHandler(logger.handlers[-1])
    logger.addHandler(TqdmLoggingHandler())
    logger.info(f'See {exp_dir} for {phase} set logs...')
    return logger


def main():
    '''
    Load pth files for dataset, generate and save crop indices for each
    '''
    args = parse_args()
    update_config(config, config_filepath=args.config, cli_options=(args.opts + ['enable_tblogging', 'False']))
    
    # Train set
    logger = setup_by_phase(config, args, 'train')
    trainset_cig = CropIndexGenerator(**config.train.dataloader.dataset)
    trainset_cig.run(logger, args.overwrite, args.debug)
    
    # Test set
    logger = setup_by_phase(config, args, 'test')
    testset_cig = CropIndexGenerator(**config.test.dataloader.dataset)
    testset_cig.run(logger, args.overwrite, args.debug)
            


if __name__ == '__main__':
    main()