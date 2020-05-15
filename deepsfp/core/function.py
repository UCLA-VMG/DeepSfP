# ------------------------------------------------------------------------------
# Copyright (c) Yunhao Ba
# Licensed under the MIT License.
# Written by Yunhao Ba (yhba@ucla.edu) Alex Gilbert (alexrgilbert@ucla.edu),
# Franklin Wang (franklinxzw@gmail.com)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
from .utils import AverageMeter, Timer, normalize_normals, \
                    split_into_patches, combine_patches

logger = logging.getLogger(__name__)


def train(dataloader, net, criterion, metric, optimizer, lr_scheduler, 
          device, epoch, writer, disp_epoch=False, print_epoch=False):
    tic = Timer()
    proc_time  = AverageMeter()
    record_time = AverageMeter()
    data_time = AverageMeter()
    errors = AverageMeter()
    losses = AverageMeter()

    for i, (_, _, _, sample) in enumerate(dataloader):
        x = sample['image'].to(device)
        y_est = sample['est'].to(device)
        y = sample['label'].to(device)
        mask = sample['mask'].to(device)
        data_time.update(tic.toc)  # Time data loading

        # Forward pass
        y_hat = net(x, y_est)

        # Optimizer
        loss = criterion(y_hat, y, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), len(sample))
        
        # Metric
        error = metric(y_hat, y, mask)
        errors.update(error.item(), len(sample))
        proc_time.update(tic.toc)  # Time inference + optimization

        # Visualize reconstructions
        if writer and disp_epoch:
            writer.add_images('images/train/est',normalize_normals(y_hat*mask.float()), epoch)
            writer.add_images('images/train/gt',normalize_normals(y), epoch)
            disp_epoch = False  # Only display first batch

        # Display stats
        if print_epoch:
            logger.info(f'Batch: [{epoch}][{i+1}/{len(dataloader)}] => ' \
                f'Data Time {data_time.val:.3f}s | Proc Time {proc_time.val:.3f}s | ' \
                f'Record Time {record_time.val:.3f}s | Loss {losses.val:.5f} | Error {errors.val:.5f}')
        record_time.update(tic.toc)  # Time viz + logging

    # Update scheduler
    if lr_scheduler:
        lr_scheduler.step(losses.avg)

    # Log epoch criteria/metric + display average stats
    if writer:
        writer.add_scalar('loss/train', losses.avg, epoch)
        writer.add_scalar('error/train', errors.avg, epoch)
    logger.info(f'Epoch Averages: Data Time {data_time.avg:.3f}s | Proc Time {proc_time.avg:.3f}s | '
                f'Record Time {record_time.avg:.3f}s | Loss {losses.avg:.5f} | Error {errors.avg:.5f}')

    return errors.avg


@torch.no_grad()
def inference(dataloader, net, metric, device, writer, reconstructions_saver = None, crop_cfg = {}):
    tic = Timer()
    proc_time  = AverageMeter()
    record_time = AverageMeter()
    data_time = AverageMeter()
    errors = AverageMeter()

    for i, (idx, _, obj_name, sample) in enumerate(dataloader):
        x = sample['image'].to(device)
        y_est = sample['est'].to(device)
        y = sample['label'].to(device)
        mask = sample['mask'].to(device)
        data_time.update(tic.toc)  # Time data loading

        # Forward pass (+ pre-processing)
        if crop_cfg.get('enable'):
            y_hat = []
            for roll_length in range(0, crop_cfg.roll_length, crop_cfg.gap_length):
                # Roll the image/priors and split into patches
                _x = split_into_patches(x, roll_length, crop_cfg.patch_size)
                _y_est = split_into_patches(y_est, roll_length, crop_cfg.patch_size)
                # Forward pass
                _y_hat = net(_x, _y_est)
                # Combine the predictions and append
                y_hat.append(combine_patches(_y_hat, roll_length))
            # Average predictions
            y_hat = torch.mean(torch.stack(y_hat), 0)
        else:
            y_hat = net(x, y_est)

        #Metric
        error = metric(y_hat, y, mask)
        errors.update(error.item(), len(idx))
        proc_time.update(tic.toc)  # Time inference

        idx, obj_name = idx.item(), obj_name[0]
        # Visualize reconstruction + log metric
        if writer:
            writer.add_images(f'images/test/{idx}_{obj_name}/pred', normalize_normals(y_hat*mask.float()))
            writer.add_images(f'images/test/{idx}_{obj_name}/gt', normalize_normals(y))
            writer.add_scalar('error/test', errors.val, idx)
        # Save reconstruction
        if reconstructions_saver:
            reconstructions_saver.update(idx, obj_name, y_hat.cpu().squeeze(), errors.val)
        # Display item stats
        logger.info(f'Test Item: [{i+1}/{len(dataloader)}] {obj_name} => ' \
                f'Data Time {data_time.val:.3f}s | Proc Time {proc_time.val:.3f}s | ' \
                f'Record Time {record_time.val:.3f}s | Error {errors.val:.5f}')
        record_time.update(tic.toc)  # Time viz + logging

    # Display average stats
    logger.info(f'Test Averages: Data Time {data_time.avg:.3f}s | Proc Time {proc_time.avg:.3f}s | '
                f'Record Time {record_time.avg:.3f}s | Error {errors.avg:.5f}')

    return errors.avg
