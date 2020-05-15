# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Union,List
from yacs.config import CfgNode as CN


# General experiment configuration
_C = CN(new_allowed=True)
_C.outputs_dir = 'outputs' # dir for saving model checkpoints
_C.enable_tblogging = True  # Whether to creat tensorboard log for train/test run
_C.tblogs_dir = 'tblogs'  # dir for saving tensorboard outputs
_C.gpu_flag = True  # Whether to use GPUs
_C.deterministic = False  # Whether to set PyTorch seed & make deterministic
_C.print_freq = 1  # Frequency (in batches) of print outs during each training epoch, 0 => no per-batch error/loss outputs during train (final epoch results still printed)
_C.disp_freq = 10  # Frequency (in epochs) of saving (a subset of) train images to TensorBoard, 0 => no images added to tb
_C.save_freq = 10  # Frequency (in epochs) of intermediate checkpoint saving, 0 => no intermediate checkpoints (final checkpoint will still be saved).

# Cudnn related params (overridden if _C.DETERMINISTIC set to True)
_C.cudnn = CN(new_allowed=True)
_C.cudnn.benchmark = True
_C.cudnn.deterministic = False
_C.cudnn.enabled = True

# Configuration for network
_C.model = CN(new_allowed=True)
_C.model.name = 'SfPNet'
# Configuration for network's weight initializer
_C.model.initializer = CN(new_allowed=True)
# Configuration for network's sub-components
_C.model.blocks = CN(new_allowed=True)
# Configuration for network's encoder sub-component
_C.model.blocks.encoder = CN(new_allowed=True)
_C.model.blocks.encoder.name = 'SfPEncoderBlock'
# Configuration for network's encoder sub-component's activation
_C.model.blocks.encoder.activation = CN(new_allowed=True)
# Configuration for network's encoder sub-component's normalization
_C.model.blocks.encoder.normalization = CN(new_allowed=True)
# Configuration for network's decoder sub-component
_C.model.blocks.decoder = CN(new_allowed=True)
_C.model.blocks.decoder.name = 'SfPDecoderBlock'
# Configuration for network's decoder sub-component's activation
_C.model.blocks.decoder.activation = CN(new_allowed=True)
# Configuration for network's decoder sub-component's normalization
_C.model.blocks.decoder.normalization = CN(new_allowed=True)
# Configuration for network's head sub-component
_C.model.blocks.head = CN(new_allowed=True)
_C.model.blocks.head.name = 'SfPHeadBlock'
# Configuration for network's head sub-component's activation
_C.model.blocks.head.activation = CN(new_allowed=True)
# Configuration for network's head sub-component's normalization
_C.model.blocks.head.normalization = CN(new_allowed=True)


# Configuration for training runs
_C.train = CN(new_allowed=True)
_C.train.end_epoch = 400
_C.train.load = False
_C.train.resume = False
_C.train.checkpoint = ''
# Configuration for training dataloader
_C.train.dataloader = CN(new_allowed=True)
_C.train.dataloader.batch_size = 16
_C.train.dataloader.shuffle = True
_C.train.dataloader.num_workers = 4
_C.train.dataloader.pin_memory = True
# Configuration for training dataloader's dataset
_C.train.dataloader.dataset = CN(new_allowed=True)
_C.train.dataloader.dataset.name = 'SurfaceNormals'
_C.train.dataloader.dataset.root = 'data'
_C.train.dataloader.dataset.data_list = 'train_list.csv'
_C.train.dataloader.dataset.precision = 'float32'
# Configuration for training dataloader's dataset's transforms (Sub-nodes correspond
# with the transfrom class names, values corresponding to the keyword args)
_C.train.dataloader.dataset.transforms = CN(new_allowed=True)
# Configuration for training optimizer
_C.train.optimizer = CN(new_allowed=True)
_C.train.optimizer.name = 'SGD'
_C.train.optimizer.lr = 1e-2
# Configuration for training learning-rate scheduler
_C.train.lr_scheduler = CN(new_allowed=True)
_C.train.lr_scheduler.name = ''
# Configuration for training loss function
_C.train.loss = CN(new_allowed=True)
_C.train.loss.name = 'MSELoss'
_C.train.loss.reduction = 'sum'
# Configuration for error metric
_C.train.metric = CN(new_allowed=True)
_C.train.metric.name = 'MeanAngularError'
_C.train.metric.mask_flag = True  # Whether to ignore background pixels in error calculation
# Configuration for testing runs
_C.test = CN(new_allowed=True)
_C.test.checkpoint = ''
_C.test.save_reconstructions = True
# Configuration for croping during test inference
_C.test.crop = CN(new_allowed=True)
# Configuration for testing dataloader
_C.test.dataloader = CN(new_allowed=True)
_C.test.dataloader.batch_size = 1
_C.test.dataloader.shuffle = False
_C.test.dataloader.num_workers = 4
_C.test.dataloader.pin_memory = True
# Configuration for testing dataloader's dataset
_C.test.dataloader.dataset = CN(new_allowed=True)
_C.test.dataloader.dataset.name = 'SurfaceNormals'
_C.test.dataloader.dataset.root = 'data'
_C.test.dataloader.dataset.data_list = 'test_list.csv'
_C.test.dataloader.dataset.precision = 'float32'
# Configuration for testing dataloader's dataset's transforms (Sub-nodes correspond
# with the transfrom class names, values corresponding to the keyword args)
_C.test.dataloader.dataset.transforms = CN(new_allowed=True)
# Configuration for error metric
_C.test.metric = CN(new_allowed=True)
_C.test.metric.name = 'MeanAngularError'
_C.test.metric.mask_flag = True  # Whether to ignore background pixels in error calculation


def update_config(cfg: CN, config_filepath: Union[os.PathLike, str] = None,
                  cli_options: List[str] = [], other_config: CN = None):
    cfg.defrost()
    # Override default configuration from file
    if config_filepath is not None:
        cfg.merge_from_file(config_filepath)
    # Override default configuration from list of key-value pairs
    if cli_options:
        cfg.merge_from_list(cli_options)
    if other_config:
        cfg.merge_from_other_cfg(other_config)
    cfg.freeze()
    return cfg.clone()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
