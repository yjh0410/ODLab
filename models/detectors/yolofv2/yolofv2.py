import torch
import torch.nn as nn

# --------------- Model components ---------------
from ...backbone import build_backbone
from ...neck import build_neck
from ...head import build_head

# --------------- External components ---------------
from utils.misc import multiclass_nms


# ------------------------ You Only Look One-level Feature ------------------------
class YOLOFv2(nn.Module):
    def __init__(self, 
                 device, 
                 cfg,
                 num_classes  :int   = 80, 
                 conf_thresh  :float = 0.05,
                 nms_thresh   :float = 0.6,
                 topk         :int   = 1000,
                 trainable    :bool  = False,
                 use_nms      :bool  = True,
                 ca_nms       :bool  = False,
                 use_aux_head: bool = False):
        super(YOLOFv2, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.cfg = cfg
        self.device = device
        self.trainable = trainable
        self.topk = topk
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.use_nms = use_nms
        self.ca_nms = ca_nms
        self.use_aux_head = use_aux_head

        # ---------------------- Network Parameters ----------------------
        ## Backbone
        self.backbone, feat_dims = build_backbone(cfg, trainable&cfg['pretrained'])

        ## Neck
        self.neck = build_neck(cfg, feat_dims[-1], cfg['head_dim'])
        
        ## Head
        self.head = build_head(cfg, cfg['head_dim'], cfg['head_dim'], num_classes)

        ## Aux Head
        if use_aux_head:
            aux_head_cfg = cfg['aux_head']
            aux_head_cfg['head_dim'] = cfg['head_dim']
            aux_head_cfg['out_stride'] = cfg['out_stride']
            self.aux_head = build_head(aux_head_cfg, aux_head_cfg['head_dim'], aux_head_cfg['head_dim'], num_classes)
            # share weight between Head and Aux-Head
            self.aux_head.cls_pred = self.head.cls_pred
            self.aux_head.reg_pred = self.head.reg_pred
            self.aux_head.obj_pred = self.head.obj_pred

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
        if self.use_nms:
            scores, labels, bboxes = multiclass_nms(
                scores, labels, bboxes, self.nms_thresh, self.num_classes, self.ca_nms)

        return bboxes, scores, labels

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
