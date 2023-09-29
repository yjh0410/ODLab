# Plain-DETR

import torch
import torch.nn as nn

from ...backbone import build_backbone
from ...transformer import build_transformer


class PlainDETR(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes :int   = 20, 
                 conf_thresh :float = 0.05,
                 topk        :int   = 100,
                 trainable   :bool  = False):
        super(PlainDETR, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.cfg = cfg
        self.device = device
        self.topk = topk
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.stride = cfg['out_stride']

        # ---------------------- Network Parameters ----------------------
        ## Backbone
        self.backbone, self.feat_dims = build_backbone(cfg, trainable&cfg['pretrained'])

        ## Transformer
        self.encoder = build_transformer(cfg, num_classes, return_intermediate=trainable)
        
