import torch
import torch.nn as nn

from ...backbone import build_backbone
from ...transformer import build_transformer


# Enhanced DETR
class DETRX(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes :int   = 80, 
                 topk        :int   = 100,
                 trainable   :bool  = False,
                 aux_loss    :bool  = False):
        super(DETRX, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.cfg = cfg
        self.device = device
        self.num_topk = topk
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.trainable = trainable
        self.max_stride = cfg['max_stride']
        self.out_stride = cfg['out_stride']

        # ---------------------- Network Parameters ----------------------
        ## Backbone
        self.backbone, self.feat_dims = build_backbone(cfg, trainable&cfg['pretrained'])
        self.input_proj = nn.Sequential(
            nn.Conv2d(self.feat_dims[-1], cfg['d_model'], kernel_size=1),
            nn.GroupNorm(32, cfg['d_model'])
        )

        ## Transformer
        self.transformer = build_transformer(cfg, num_classes, return_intermediate=trainable)

    def decode_bboxes(self, reg_preds, bbox_encode=False):
        if not bbox_encode:
            box_preds_x1y1 = reg_preds[..., :2] - 0.5 * reg_preds[..., 2:]
            box_preds_x2y2 = reg_preds[..., :2] + 0.5 * reg_preds[..., 2:]
            box_preds = torch.cat([box_preds_x1y1, box_preds_x2y2], dim=-1)

        return box_preds
    
    def post_process(self, cls_pred, box_pred):
        ## Top-k select
        cls_pred = cls_pred[0].flatten().sigmoid_()
        box_pred = box_pred[0]
        predicted_prob, topk_idxs = cls_pred.sort(descending=True)
        topk_idxs = topk_idxs[:self.num_topk]
        topk_box_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
        ## Top-k results
        topk_scores = predicted_prob[:self.num_topk]
        topk_labels = topk_idxs % self.num_classes
        topk_bboxes = box_pred[topk_box_idxs]

        return topk_bboxes, topk_scores, topk_labels

    @torch.no_grad()
    def inference_single_image(self, x):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(x)
        feat = self.input_proj(pyramid_feats[-1])

        # ---------------- Transformer ----------------
        outputs = self.transformer(src=feat)

        # ---------------- PostProcess ----------------
        cls_preds = outputs["pred_logits"]
        box_preds = self.decode_bboxes(outputs["pred_boxes"])
        bboxes, scores, labels = self.post_process(cls_preds, box_preds)

        return bboxes, scores, labels

    def forward(self, x, mask=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # ---------------- Backbone ----------------
            pyramid_feats = self.backbone(x)
            feat = self.input_proj(pyramid_feats[-1])

            # ---------------- Transformer ----------------
            outputs = self.transformer(src=feat, is_train=True, src_mask=mask)
            
            return outputs
