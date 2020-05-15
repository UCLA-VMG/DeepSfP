# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from typing import Union, List
import os
import logging
from pprint import pformat
from datetime import datetime
import shutil
from git import Repo
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
import torch.backends.cudnn as cudnn


def SfParser(description: str = None, add_help: bool = True):
    parser = argparse.ArgumentParser(description=description, add_help=add_help)

    parser.add_argument('--config','-c', help='experiment configuration filename', type=str)
    parser.add_argument('opts',help='Additional config override options, e.g. ["DATASET.TRAINSET_FILE", \
                        "new_train_list.csv", "DATASET.TRANSFORMS.RANDOM_CROP.BOUNDS", "(256, 256)"]',
                        nargs='*')
    return parser


def setup_experiment(cfg, cfg_name: str = None, root_dir: Union[str, os.PathLike] = '.', 
                    phase: str = 'train', name: str = None, quiet: bool = False, meta: List = []):

    # Create experiment directory
    if not cfg_name:
        cfg_name = 'defaults'
    time_str = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    if not name:
        name = phase
    exp_id = f'{os.path.basename(cfg_name).split(".")[0]}_{time_str}_{name}'
    experiment_dir = os.path.realpath(os.path.join(root_dir, cfg.outputs_dir, 
                        cfg.get(phase).dataloader.dataset.name, exp_id))
    os.makedirs(experiment_dir, exist_ok=True)

    # Save frozen configuration
    cfg_copy = os.path.join(experiment_dir, f'{exp_id}.yaml')
    with open(cfg_copy,'w+') as cfg_copy:
        cfg.dump(stream=cfg_copy)

    # Save train/test splits
    datacfg = cfg.get(phase).dataloader.dataset
    dataroot, dataset, datafile = datacfg.root, datacfg.name, datacfg.data_list
    datafile_copy = os.path.join(experiment_dir, f'{phase}_set.csv')
    shutil.copy(os.path.join(root_dir, dataroot, dataset, datafile), datafile_copy)

    # Create log file
    logger = logging.getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    logging.basicConfig(filename=os.path.join(experiment_dir, f'{exp_id}.log'),
                        format='%(asctime)-15s %(message)s')
    logger.setLevel(logging.INFO)
    if not quiet:
        console = logging.StreamHandler()
        logger.addHandler(console)
    logger.info(f'=> Experiment directory located at: {os.path.realpath(experiment_dir)}...')

    # Log current git hash
    repo = Repo(search_parent_directories=True)
    msg = f'[{repo.remotes[0].url}] -> {repo.working_dir} @ {repo.head.object.hexsha}'
    if repo.is_dirty():
        msg += f' (UNCOMMITTED CHANGES)'
    logger.info(msg)

    # Create tensorboard directory
    tensorboard_log_dir = None
    if cfg.enable_tblogging:
        tensorboard_log_dir = os.path.realpath(os.path.join(root_dir,cfg.tblogs_dir,dataset,exp_id,phase))
        logger.info(f'=> creating {os.path.realpath(tensorboard_log_dir)}')
        os.makedirs(tensorboard_log_dir, exist_ok=True)

    # Configure GPU and Determinism
    gpu_flag = torch.cuda.is_available() and cfg.gpu_flag
    device = torch.device('cuda' if gpu_flag else 'cpu')
    cudnn.enabled = gpu_flag and cfg.cudnn.enabled
    if cfg.deterministic:
        torch.manual_seed(0)
        np.random.seed(0)
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = cfg.cudnn.benchmark
        cudnn.determinstic = cfg.cudnn.benchmark
    
    # Log provided metadata
    for msg in meta:
        logger.info(pformat(msg))
        
    return logger, experiment_dir, tensorboard_log_dir, exp_id, device


def load_checkpoint(checkpoint: Union[str, os.PathLike], device: torch.device, 
                    model: nn.Module, logger: logging.Logger, resume: bool = False, 
                    optimizer: Optimizer = None, lr_scheduler: object = None, 
                    end_epoch: int = 0):
    model_state_file = os.path.realpath(checkpoint)
    logger.info(f'=> Loading checkpoint @ {model_state_file}...')
    checkpoint = torch.load(model_state_file, map_location=device)
    logger.info(f'=> Loading model state from checkpoint...')
    model.load_state_dict(checkpoint['state_dict'])
    if resume:
        start_epoch = checkpoint['epoch'] + 1
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logger.info(f"=> Resuming training from epoch {checkpoint['epoch']}...")
        if end_epoch < start_epoch:
            raise ValueError(f'Attempted to resume training for 0 additional epochs! '
                            'Please increase "TRAIN.END_EPOCH" or disable "TRAIN.RESUME"!')


def save_checkpoint(model: nn.Module, optimizer: Optimizer, lr_scheduler: object, epoch: int, 
                    mae: float, output_dir: Union[str,os.PathLike], logger: logging.Logger, 
                    filename: Union[str,os.PathLike] = 'checkpoint.pth'):
    ckpt_pth = os.path.realpath(os.path.join(output_dir, filename))
    logger.info(f'=> saving checkpoint to {ckpt_pth}')
    torch.save({"state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "lr_scheduler": lr_scheduler.state_dict(),
                 "epoch": epoch, "mae": mae}, ckpt_pth)

    # Update pointer to latest checkpoint
    latest_pth = os.path.join(os.path.realpath(output_dir), 'latest.pth')
    if os.path.islink(latest_pth):
        os.remove(latest_pth)
    os.symlink(ckpt_pth, latest_pth)


class ReconstructionsSaver(object):
    def __init__(self, save_dir: Union[str, os.PathLike], dataset: str,
                save_reconstructions: bool = False, logger: logging.Logger = None):
        self._save_dir = save_dir
        self._dataset = dataset
        self._save_reconstructions = save_reconstructions
        self._reconstructions = {}
        self._results_idx = []
        self._results_data = []

        self._logger = logger

    def _name_to_dict(self, obj_name):
        if self._dataset == 'SurfaceNormals':
            obj_name_list = obj_name.split('_')
            light_idx = 2 if obj_name_list[1] in ['sunny','cloudy'] else 1
            light, obj, dir = obj_name_list[:light_idx], '_'.join(obj_name_list[light_idx:-1]), obj_name_list[-1]
            return {'lighting': light, 'object': obj, 'orientation': dir}
        else:
            raise NotImplemented(f'Failed to map object name! '
                                 f'{self._dataset} dataset does not exist.'
                                 f' Must choose from: {DATASETS}')

    def update(self, idx, obj_name, y_hat, error):
        # Results
        self._results_idx.append(idx)
        data_dict = self._name_to_dict(obj_name)
        data_dict.update({'error': error})
        self._results_data.append(data_dict)
        # Reconstructions
        if self._save_reconstructions:
            self._reconstructions[obj_name] = {'reconstruction': y_hat, 'error': error}

    def save(self):
        res_pth = os.path.join(self._save_dir, 'results.csv')
        if self._logger:
            self._logger.info(f'Saving test results dataframe to {res_pth}...')
        res_df =pd.DataFrame(data=self._results_data, 
                             index=pd.Index(self._results_idx, name='obj_idx'))
        res_df.to_csv(res_pth)
        if self._save_reconstructions:
            rec_pth = os.path.join(self._save_dir, 'reconstructions.pth')
            if self._logger:
                self._logger.info(f'Saving test reconstructions to {rec_pth}...')
            torch.save(self._reconstructions, rec_pth)
