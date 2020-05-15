# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deepsfp.models import model_factory
from deepsfp.config import config, update_config
from deepsfp.datasets import dataloader_factory
from deepsfp.core import metric_factory, inference
from deepsfp.utils import SfParser, setup_experiment, \
                        load_checkpoint, ReconstructionsSaver


def main():
    args = SfParser(description='Test DeepSfP').parse_args()

    update_config(config, config_filepath=args.config, cli_options=args.opts)

    logger, exp_dir, tblogs_dir, _, device = \
        setup_experiment(config, args.config, phase='test', meta=[args, config])

    # Dataloader
    dataloader = dataloader_factory.build(**config.test.dataloader)            

    # Metric
    metric = metric_factory.build(**config.test.metric).to(device)
    
    # Model
    model = model_factory.build(logger=logger, **config.model).to(device)
    load_checkpoint(config.test.checkpoint, device, model, logger)
    model.eval()

    # TensorBoard logging
    writer = SummaryWriter(tblogs_dir) if tblogs_dir else None

    # Reconstructions saving
    save_preds = config.test.save_reconstructions
    dataset = dataloader.dataset.__class__.__name__
    predictions = ReconstructionsSaver(exp_dir, dataset, save_preds, logger)

    # Run inference...
    inference(dataloader, model, metric, device, writer, predictions, config.test.crop)

    # Save results (and reconstructions, if specified in config)
    predictions.save()

    # Cleanup TB logger
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
