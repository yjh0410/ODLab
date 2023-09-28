import numpy as np
import math
import torch
import torch.nn as nn

# --------------- Model components ---------------
from ...modules.backbone import build_backbone
from ...modules.neck import build_neck
from ...modules.head import build_head

# --------------- External components ---------------
from utils.misc import multiclass_nms


# ------------------------ You Only Look One-level Feature ------------------------
class YOLOF(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes :int   = 20, 
                 conf_thresh :float = 0.05,
                 nms_thresh  :float = 0.6,
                 topk        :int   = 1000,
                 trainable   :bool  = False):
        super(YOLOF, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.cfg = cfg
        self.device = device
        self.topk = topk
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = cfg['out_stride']
        # Anchor size
        self.anchor_size = torch.as_tensor(cfg['anchor_size'])
        self.num_anchors = len(cfg['anchor_size'])

        # ---------------------- Network Parameters ----------------------
        ## Backbone
        self.backbone, self.feat_dims = build_backbone(cfg, trainable&cfg['pretrained'])

        ## Encoder
        self.encoder = build_neck(cfg, self.feat_dims[-1], cfg['d_model'])
        
        ## Decoder
        self.decoder = build_head(cfg, cfg['d_model'], cfg['d_model'], num_classes)

    def post_process(self, cls_pred, box_pred):
        """
        Input:
            cls_pred: (Tensor) [[H x W x KA, C]
            box_pred: (Tensor)  [H x W x KA, 4]
        """
        cls_pred = cls_pred[0]
        box_pred = box_pred[0]
        
        # (H x W x KA x C,)
        scores_i = cls_pred.sigmoid().flatten()

        # Keep top k top scoring indices only.
        num_topk = min(self.topk, box_pred.size(0))

        # torch.sort is actually faster than .topk (at least on GPUs)
        predicted_prob, topk_idxs = scores_i.sort(descending=True)
        topk_scores = predicted_prob[:num_topk]
        topk_idxs = topk_idxs[:num_topk]

        # filter out the proposals with low confidence score
        keep_idxs = topk_scores > self.conf_thresh
        topk_idxs = topk_idxs[keep_idxs]

        # final scores
        scores = topk_scores[keep_idxs]
        # final labels
        labels = topk_idxs % self.num_classes
        # final bboxes
        anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
        bboxes = box_pred[anchor_idxs]

        # to cpu & numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, False)

        return bboxes, scores, labels

    @torch.no_grad()
    def inference_single_image(self, x):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(x)

        # ---------------- Encoder ----------------
        en_feat = self.encoder(pyramid_feats[-1])

        # ---------------- Decoder ----------------
        outputs = self.decoder(en_feat)

        # ---------------- PostProcess ----------------
        cls_pred = outputs["pred_cls"]
        box_pred = outputs["pred_box"]
        bboxes, scores, labels = self.post_process(cls_pred, box_pred)
        # normalize bbox
        bboxes[..., 0::2] /= x.shape[-1]
        bboxes[..., 1::2] /= x.shape[-2]
        bboxes = bboxes.clip(0., 1.)

        return bboxes, scores, labels

    def forward(self, x, mask=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # ---------------- Backbone ----------------
            pyramid_feats = self.backbone(x)

            # ---------------- Encoder ----------------
            en_feat = self.encoder(pyramid_feats[-1])

            # ---------------- Decoder ----------------
            outputs = self.decoder(en_feat, mask)

            return outputs 
