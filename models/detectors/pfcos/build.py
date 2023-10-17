#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .criterion import build_criterion
from .pfcos import PlainFCOS


# build PlainFCOS
def build_pfcos(cfg, device, num_classes=80, trainable=False):
    # -------------- Build PlainFCOS --------------
    model = PlainFCOS(cfg          = cfg,
                      num_classes  = num_classes,
                      conf_thresh  = cfg['train_conf_thresh'] if trainable else cfg['test_conf_thresh'],
                      nms_thresh   = cfg['train_nms_thresh']  if trainable else cfg['test_nms_thresh'],
                      topk         = cfg['train_topk']        if trainable else cfg['test_topk'],
                      trainable    = trainable,
                      ca_nms       = False if trainable else cfg['nms_class_agnostic'],
                      use_aux_head = cfg['use_aux_head'])

    # -------------- Build Criterion --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, num_classes)

    return model, criterion