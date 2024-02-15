#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .criterion import build_criterion
from .plain_detr import PlainDETR


# build object detector
def build_plain_detr(cfg, device, num_classes=80, trainable=False):
    # -------------- Build RT-DETR --------------
    model = PlainDETR(cfg         = cfg,
                      num_classes = num_classes,
                      conf_thresh = cfg['train_conf_thresh'] if trainable else cfg['test_conf_thresh'],
                      nms_thresh  = cfg['train_nms_thresh']  if trainable else cfg['test_nms_thresh'],
                      topk        = cfg['train_topk']        if trainable else cfg['test_topk'],
                      )
            
    # -------------- Build criterion --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, num_classes, aux_loss=True)
        
    return model, criterion