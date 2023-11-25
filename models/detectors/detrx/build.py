#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .criterion import build_criterion
from .detrx import DETRX


# build DETRX
def build_detrx(cfg, device, num_classes=80, trainable=False):
    # -------------- Build DETRX --------------
    model = DETRX(cfg         = cfg,
                  device      = device,
                  num_classes = num_classes,
                  topk        = cfg['train_topk'] if trainable else cfg['test_topk'],
                  trainable   = trainable,
                  aux_loss    = trainable)
            
    # -------------- Build Criterion --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, num_classes, aux_loss=True)

    return model, criterion