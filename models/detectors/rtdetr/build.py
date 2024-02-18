#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .criterion import build_criterion
from .rtdetr import RTDETR


# build object detector
def build_rtdetr(cfg, device, num_classes=80, trainable=False):
    # -------------- Build RT-DETR --------------
    model = RTDETR(cfg         = cfg,
                   num_classes = num_classes,
                   conf_thresh = cfg['train_conf_thresh'] if trainable else cfg['test_conf_thresh'],
                   nms_thresh  = cfg['train_nms_thresh']  if trainable else cfg['test_nms_thresh'],
                   topk        = cfg['train_topk']        if trainable else cfg['test_topk'],
                   use_nms     = False,
                   )
            
    # -------------- Build criterion --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, num_classes)
        
    return model, criterion