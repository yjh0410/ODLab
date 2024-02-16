import torch
import torch.nn as nn
import random

from ...backbone          import build_backbone
from ...neck              import build_neck
from ...basic.transformer import RTDETRTransformer

from utils.misc import multiclass_nms


# Real-time DETR
class RTDETR(nn.Module):
    def __init__(self,
                 cfg,
                 num_classes = 80,
                 conf_thresh = 0.1,
                 nms_thresh  = 0.5,
                 topk        = 300,
                 use_nms     = False,
                 ca_nms      = False,
                 ):
        super().__init__()
        # ----------- Basic setting -----------
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_topk = topk
        ## Post-process parameters
        self.ca_nms = ca_nms
        self.use_nms = use_nms
        self.num_topk = topk
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh

        # ----------- Network setting -----------
        ## Image encoder: Backbone
        self.backbone, feat_dims = build_backbone(cfg)

        ## Image encoder: Hybrid Encoder
        self.hybrid_encoder = build_neck(cfg, feat_dims, cfg['hidden_dim'])

        ## Transformer
        self.detect_decoder = RTDETRTransformer(in_dims             = [cfg['hidden_dim']] * 3,
                                                hidden_dim          = cfg['hidden_dim'],
                                                strides             = cfg['out_stride'],
                                                num_classes         = num_classes,
                                                num_queries         = cfg['num_queries'],
                                                num_heads           = cfg['de_num_heads'],
                                                num_layers          = cfg['de_num_layers'],
                                                num_levels          = 3,
                                                num_points          = cfg['de_num_points'],
                                                ffn_dim             = cfg['de_ffn_dim'],
                                                dropout             = cfg['de_dropout'],
                                                act_type            = cfg['de_act'],
                                                pre_norm            = cfg['de_pre_norm'],
                                                return_intermediate = True,
                                                num_denoising       = cfg['dn_num_denoising'],
                                                label_noise_ratio   = cfg['dn_label_noise_ratio'],
                                                box_noise_scale     = cfg['dn_box_noise_scale'],
                                                learnt_init_query   = cfg['learnt_init_query'],
                                                )

    def deploy(self):
        assert not self.training
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 

    def post_process(self, box_pred, cls_pred):
        # xywh -> xyxy
        box_preds_x1y1 = box_pred[..., :2] - 0.5 * box_pred[..., 2:]
        box_preds_x2y2 = box_pred[..., :2] + 0.5 * box_pred[..., 2:]
        box_pred = torch.cat([box_preds_x1y1, box_preds_x2y2], dim=-1)
        
        # Top-k select
        cls_pred = cls_pred[0].flatten().sigmoid_()
        box_pred = box_pred[0]

        # Keep top k top scoring indices only.
        num_topk = min(self.num_topk, box_pred.size(0))

        # Topk candidates
        predicted_prob, topk_idxs = cls_pred.sort(descending=True)
        topk_scores = predicted_prob[:num_topk]
        topk_idxs = topk_idxs[:self.num_topk]

        # Filter out the proposals with low confidence score
        keep_idxs = topk_scores > self.conf_thresh
        topk_scores = topk_scores[keep_idxs]
        topk_idxs = topk_idxs[keep_idxs]
        topk_box_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')

        ## Top-k results
        topk_labels = topk_idxs % self.num_classes
        topk_bboxes = box_pred[topk_box_idxs]

        topk_scores = topk_scores.cpu().numpy()
        topk_labels = topk_labels.cpu().numpy()
        topk_bboxes = topk_bboxes.cpu().numpy()

        # nms
        if self.use_nms:
            topk_scores, topk_labels, topk_bboxes = multiclass_nms(
                topk_scores, topk_labels, topk_bboxes, self.nms_thresh, self.num_classes, self.nms_class_agnostic)

        return topk_bboxes, topk_scores, topk_labels
    
    def forward(self, src, src_mask=None, targets=None):
        if self.training:
            sz = random.choice(self.cfg['random_size'])
            src = nn.functional.interpolate(src, size=[sz, sz])

        # ----------- Image Encoder -----------
        pyramid_feats = self.backbone(src)
        pyramid_feats = self.hybrid_encoder(pyramid_feats)

        # ----------- Transformer -----------
        outputs = self.detect_decoder(pyramid_feats, targets)

        if not self.training:
            box_pred = outputs["pred_boxes"]
            cls_pred = outputs["pred_logits"]
            
            # post-process
            bboxes, scores, labels = self.post_process(box_pred, cls_pred)

            return bboxes, scores, labels

        return outputs
