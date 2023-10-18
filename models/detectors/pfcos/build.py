#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .criterion import build_criterion
from .pfcos import PlainFCOS


# build PlainFCOS
def build_pfcos(cfg, device, num_classes=80, trainable=False):
    # -------------- Build PlainFCOS --------------
    model = PlainFCOS(cfg         = cfg,
                      device      = device,
                      num_classes = num_classes,
                      topk        = cfg['train_topk'] if trainable else cfg['test_topk'],
                      trainable   = trainable
                      )
            
    # -------------- Build Criterion --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, device, num_classes)

    return model, criterion