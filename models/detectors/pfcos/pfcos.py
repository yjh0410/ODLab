import torch
import torch.nn as nn

# --------------- Model components ---------------
from ...backbone import build_backbone
from ...neck import build_neck
from ...head import build_head


# ------------------------ Plain FCOS ------------------------
class PlainFCOS(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes :int   = 80, 
                 topk        :int   = 1000,
                 trainable   :bool  = False,
                 aux_loss    :bool  = False):
        super(PlainFCOS, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.cfg = cfg
        self.device = device
        self.num_topk = topk
        self.trainable = trainable
        self.aux_loss = aux_loss
        self.num_classes = num_classes

        # ---------------------- Network Parameters ----------------------
        ## Backbone
        self.backbone, feat_dims = build_backbone(cfg, trainable&cfg['pretrained'])

        ## Neck
        self.neck = build_neck(cfg, feat_dims[-1], cfg['head_dim'])
        
        ## Heads
        self.head = build_head(cfg, cfg['head_dim'], cfg['head_dim'], num_classes)

    def post_process(self, cls_pred, box_pred):
        """
            Inputs:
                cls_pred: (Tensor) [B, Nq, Nc], where B should be 1.
                box_pred: (Tensor) [B, Nq, 4], where B should be 1.
            Outputs:
                topk_bboxes: (Tensor) [Topk, 4]
                topk_scores: (Tensor) [Topk,]
                topk_labels: (Tensor) [Topk,]
        """
        cls_pred = cls_pred[0].flatten().sigmoid_()
        box_pred = box_pred[0]
        ## Top-k select
        predicted_prob, topk_idxs = cls_pred.sort(descending=True)
        topk_idxs = topk_idxs[:self.num_topk]
        topk_box_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
        ## Top-k results
        topk_scores = predicted_prob[:self.num_topk]
        topk_labels = topk_idxs % self.num_classes
        topk_bboxes = box_pred[topk_box_idxs]

        return topk_bboxes, topk_scores, topk_labels

    @torch.jit.unused
    def set_aux_loss(self, outputs_class, output_delta, outputs_coord, ref_points):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_deltas': b, 'pred_boxes': c, 'ref_points': d}
                for a, b, c, d in zip(outputs_class[:-1], output_delta[:-1], outputs_coord[:-1], ref_points[:-1])]

    @torch.no_grad()
    def inference_single_image(self, x):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(x)

        # ---------------- Neck ----------------
        feat = self.neck(pyramid_feats[-1])

        # ---------------- Heads ----------------
        output_classes, output_coords, _, _ = self.head(feat)

        # ---------------- PostProcess ----------------
        cls_pred = output_classes[-1]
        box_pred = output_coords[-1]
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

            # ---------------- Heads ----------------
            output_classes, output_coords, output_deltas, ref_points = self.head(feat, mask)
            outputs = {'pred_logits': output_classes[-1],
                       'pred_deltas': output_deltas[-1],
                       'pred_boxes': output_coords[-1],
                       'ref_points': ref_points[-1]}
            if self.aux_loss:
                outputs['aux_outputs'] = self.set_aux_loss(output_classes, output_deltas, output_coords, ref_points)
            
            # Reshape mask
            if mask is not None:
                B, _, H, W = feat.size()
                # [B, H, W]
                mask = torch.nn.functional.interpolate(mask[None].float(), size=[H, W]).bool()[0]
                # [B, H, W] -> [B, M]
                mask = mask.flatten(1)
            outputs['masks'] = mask

            return outputs
