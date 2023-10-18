import torch
import torch.nn as nn

# --------------- Model components ---------------
from ...backbone import build_backbone
from ...neck import build_neck
from ...head import build_head

# --------------- External components ---------------
from utils.misc import multiclass_nms


# ------------------------ Plain Fully Convolutional One-Stage Detector ------------------------
class PlainFCOS(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes  :int   = 80, 
                 topk         :int   = 1000,
                 trainable    :bool  = False,
                 use_aux_head :bool = False):
        super(PlainFCOS, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.cfg = cfg
        self.device = device
        self.trainable = trainable
        self.topk = topk
        self.num_classes = num_classes
        self.use_aux_head = use_aux_head

        # ---------------------- Network Parameters ----------------------
        ## Backbone
        self.backbone, feat_dims = build_backbone(cfg, trainable&cfg['pretrained'])

        ## Neck
        self.neck = build_neck(cfg, feat_dims[-1], cfg['head_dim'])
        
        ## Head
        self.head = build_head(cfg, cfg['head_dim'], cfg['head_dim'], num_classes)

        ## Aux-Head
        if use_aux_head:
            aux_head_cfg = cfg['aux_head']
            self.aux_head = build_head(aux_head_cfg, aux_head_cfg['head_dim'], aux_head_cfg['head_dim'], num_classes)

    def post_process(self, cls_pred, box_pred):
        """
        Input:
            cls_pred: (Tensor) [B, H x W, C]
            box_pred: (Tensor) [B, H x W, 4]
        """        
        cls_pred = cls_pred[0]
        box_pred = box_pred[0]
        
        # (H x W x C,)
        cls_scores = cls_pred.sigmoid().flatten()

        # Keep top k top scoring indices only.
        num_topk = min(self.topk, box_pred.size(0))

        # torch.sort is actually faster than .topk (at least on GPUs)
        predicted_prob, topk_idxs = cls_scores.sort(descending=True)

        # final scores
        topk_scores = predicted_prob[:num_topk]
        topk_idxs = topk_idxs[:num_topk]

        # final labels
        topk_labels = topk_idxs % self.num_classes

        # final bboxes
        anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
        topk_bboxes = box_pred[anchor_idxs]

        # to cpu & numpy
        topk_scores = topk_scores.cpu().numpy()
        topk_labels = topk_labels.cpu().numpy()
        topk_bboxes = topk_bboxes.cpu().numpy()

        return topk_bboxes, topk_scores, topk_labels

    @torch.no_grad()
    def inference_single_image(self, x):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(x)

        # ---------------- Neck ----------------
        feat = self.neck(pyramid_feats[-1])

        # ---------------- Heads ----------------
        outputs = self.head(feat)

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

            # ---------------- Neck ----------------
            feat = self.neck(pyramid_feats[-1])

            # ---------------- Head ----------------
            outputs = self.head(feat, mask)

            # ---------------- Aux Head ----------------
            if self.use_aux_head:
                aux_outputs = self.aux_head(feat, mask)
                outputs['aux_outputs'] = aux_outputs

            return outputs 
