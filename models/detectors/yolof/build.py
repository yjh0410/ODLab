#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .criterion import build_criterion
from .yolof import YOLOF


# build YOLOF
def build_yolof(cfg, device, num_classes=80, trainable=False):
    # -------------- Build YOLOF --------------
    model = YOLOF(cfg         = cfg,
                  device      = device,
                  num_classes = num_classes,
                  conf_thresh = cfg['train_conf_thresh'] if trainable else cfg['test_conf_thresh'],
                  nms_thresh  = cfg['train_nms_thresh']  if trainable else cfg['test_nms_thresh'],
                  topk        = cfg['train_topk']        if trainable else cfg['test_topk'],
                  trainable   = trainable)
            
    # -------------- Build Criterion --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, device, num_classes)

    return model, criterion