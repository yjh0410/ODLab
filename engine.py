# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import time
import sys
from typing import Iterable

import torch
from utils import distributed_utils
from utils.vis_tools import vis_data


def train_one_epoch(cfg,
                    model       : torch.nn.Module,
                    criterion   : torch.nn.Module,
                    data_loader : Iterable,
                    optimizer   : torch.optim.Optimizer,
                    device      : torch.device,
                    epoch       : int,
                    max_epoch   : int,
                    max_norm    : float,
                    vis_target  : bool,
                    warmup_lr_scheduler,
                    ):
    model.train()
    criterion.train()
    lr_warmup_stage = True
    epoch_size = len(data_loader)

    t0 = time.time()
    for iter_i, (images, masks, targets) in enumerate(data_loader):
        ni = iter_i + epoch * epoch_size
        # WarmUp
        if ni < cfg['warmup_iters'] and lr_warmup_stage:
            warmup_lr_scheduler(ni, optimizer)
        elif ni == cfg['warmup_iters'] and lr_warmup_stage:
            print('Warmup stage is over.')
            lr_warmup_stage = False
            warmup_lr_scheduler.set_lr(optimizer, cfg['base_lr'], cfg['base_lr'])

        # To device
        images = images.to(device)
        masks  = masks.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Visualize train targets
        if vis_target:
            vis_data(images, targets)

        # Inference
        outputs = model(images, masks)

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

        # Logs
        if distributed_utils.is_main_process() and iter_i % 10 == 0:
            t1 = time.time()
            cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
            cur_lr_dict = {'lr': cur_lr[0], 'lr_bk': cur_lr[1]}
            # basic infor
            log =  '[Epoch: {}/{}]'.format(epoch+1, max_epoch)
            log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
            log += '[lr: {:.6f}][lr_bk: {:.6f}]'.format(cur_lr_dict['lr'], cur_lr_dict['lr_bk'])
            # loss infor
            for k in loss_dict_reduced.keys():
                log += '[{}: {:.2f}]'.format(k, loss_dict_reduced[k])
            # other infor
            log += '[time: {:.2f}]'.format(t1 - t0)

            # print log infor
            print(log, flush=True)
            
            t0 = time.time()

    return
