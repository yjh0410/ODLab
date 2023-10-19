#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .criterion import build_criterion
from .yolofv2 import YOLOFv2


# build YOLOFv2
def build_yolofv2(cfg, device, num_classes=80, trainable=False):
    # -------------- Build YOLOF --------------
    model = YOLOFv2(cfg         = cfg,
                    device      = device,
                    num_classes = num_classes,
                    conf_thresh = cfg['train_conf_thresh'] if trainable else cfg['test_conf_thresh'],
                    nms_thresh  = cfg['train_nms_thresh']  if trainable else cfg['test_nms_thresh'],
                    topk        = cfg['train_topk']        if trainable else cfg['test_topk'],
                    trainable   = trainable,
                    use_nms     = cfg['use_nms'],
                    ca_nms      = False if trainable else cfg['nms_class_agnostic'],
                    use_aux_head= cfg['use_aux_head'] & trainable)
            
    # -------------- Build Criterion --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, device, num_classes)

    return model, criterion