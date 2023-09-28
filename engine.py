# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import utils.misc as utils
from utils import distributed_utils


def train_one_epoch(cfg,
                    model       : torch.nn.Module,
                    criterion   : torch.nn.Module,
                    data_loader : Iterable,
                    optimizer   : torch.optim.Optimizer,
                    device      : torch.device,
                    epoch       : int,
                    warmup_lr_scheduler,
                    max_norm    : float = 0,
                    ):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    lr_warmup_stage = True
    epoch_size = len(data_loader)

    for iter_i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        ni = iter_i + epoch * epoch_size
        # WarmUp
        if ni < cfg['warmup_iter'] and lr_warmup_stage:
            warmup_lr_scheduler(ni, optimizer)
        elif ni == cfg['warmup_iter'] and lr_warmup_stage:
            print('Warmup stage is over.')
            lr_warmup_stage = False
            warmup_lr_scheduler.set_lr(optimizer, cfg['base_lr'], cfg['base_lr'])

        # To device
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Inference
        outputs = model(samples)

        # Compute loss
        loss_dict = criterion(outputs, targets)
        losses = loss_dict['losses']
        loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

        # Check loss
        if not math.isfinite(losses.item()):
            print("Loss is {}, stopping training".format(losses.item()))
            print(loss_dict_reduced)
            sys.exit(1)

        # Backward
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=losses.item(), **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
