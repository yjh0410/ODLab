#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .criterion import build_criterion
from .fcos import FCOS


# build FCOS
def build_fcos(cfg, device, num_classes=90, trainable=False):
    # -------------- Build FCOS --------------
    model = FCOS(cfg         = cfg,
                 device      = device,
                 num_classes = num_classes,
                 conf_thresh = cfg['train_conf_thresh'] if trainable else cfg['test_conf_thresh'],
                 nms_thresh  = cfg['train_nms_thresh']  if trainable else cfg['test_nms_thresh'],
                 topk        = cfg['train_topk']        if trainable else cfg['test_topk'],
                 trainable   = trainable,
                 ca_nms      = False if trainable else cfg['nms_class_agnostic'])
            
    # -------------- Build Criterion --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, device, num_classes)

    return model, criterion