# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
import os
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deepsfp.config import config, update_config
from deepsfp.models import model_factory
from deepsfp.datasets import dataloader_factory
from deepsfp.core import optimizer_factory, lr_scheduler_factory, \
                         loss_factory, metric_factory, train
from deepsfp.utils import SfParser, setup_experiment, \
                          load_checkpoint, save_checkpoint


def main():
    args = SfParser(description='Train DeepSfP').parse_args()

    update_config(config, config_filepath=args.config, cli_options=args.opts)

    logger, exp_dir, tblogs_dir, _, device = \
        setup_experiment(config, args.config, phase='train', meta=[args, config])

    # Dataloader
    dataloader = dataloader_factory.build(**config.train.dataloader)

    # Model & optimization algorithms
    start_epoch, end_epoch = 1, config.train.end_epoch
    model = model_factory.build(logger=logger, **config.model).to(device)
    optimizer = optimizer_factory.build(model, **config.train.optimizer)
    lr_scheduler = lr_scheduler_factory.build(optimizer, **config.train.lr_scheduler)
    if config.train.load:
        load_checkpoint(config.train.checkpoint, device, model, logger, config.train.resume, 
                        optimizer, lr_scheduler, end_epoch)
    model.train()

    # Loss & error functins
    criterion = loss_factory.build(**config.train.loss).to(device)
    metric = metric_factory.build(**config.train.metric).to(device)

    # TensorBoard logging
    writer = SummaryWriter(tblogs_dir) if tblogs_dir else None

    mae = 0.
    for epoch in range(start_epoch, end_epoch + 1):
        disp_epoch = config.disp_freq and (epoch % config.disp_freq) == 0
        print_epoch = config.print_freq and (epoch % config.print_freq) == 0
        save_epoch = config.save_freq and epoch % config.save_freq == 0

        mae = train(dataloader, model, criterion, metric, optimizer, lr_scheduler, 
                    device, epoch, writer, disp_epoch, print_epoch)

        if save_epoch:
            save_checkpoint(model, optimizer, lr_scheduler, epoch, mae, 
                            exp_dir, logger, f'checkpoint_{epoch}.pth')

    final_ckpt_name = f'finalckpt-{start_epoch}_{end_epoch}-{mae:.1f}.pth'.replace('.','_',1)
    save_checkpoint(model, optimizer, lr_scheduler, epoch, mae, exp_dir, logger, final_ckpt_name)

    if writer:
        writer.close()


if __name__ == '__main__':
    main()
