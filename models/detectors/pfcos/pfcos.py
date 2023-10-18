import math
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
        self.DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)
        # ---------------------- Basic Parameters ----------------------
        self.cfg = cfg
        self.device = device
        self.trainable = trainable
        self.topk = topk
        self.stride = cfg['out_stride']
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
            aux_head_cfg['head_dim'] = cfg['head_dim']
            self.aux_head = build_head(aux_head_cfg, aux_head_cfg['head_dim'], aux_head_cfg['head_dim'], num_classes)

        ## Pred layers
        self.cls_pred = nn.Conv2d(cfg['head_dim'], num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(cfg['head_dim'], 4, kernel_size=3, padding=1)
                
        # init bias
        self._init_pred_layers()

    def _init_pred_layers(self):
        for module in [self.cls_pred, self.reg_pred]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)
        # init the bias of cls pred
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        torch.nn.init.constant_(self.cls_pred.bias, bias_value)
        
    def get_anchors(self, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchors = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        anchors *= self.stride

        return anchors
        
    def decode_boxes(self, pred_reg, anchors):
        """
            pred_reg: (Tensor) [B, M, 4] or [M, 4]
            anchors:  (Tensor) [1, M, 2] or [M, 2]
        """
        pred_cxcy = anchors + pred_reg[..., :2] * self.stride
        pred_bwbh = torch.exp(pred_reg[..., 2:].clamp(max=self.DEFAULT_SCALE_CLAMP)) * self.stride

        pred_x1y1 = pred_cxcy - pred_bwbh * 0.5
        pred_x2y2 = pred_cxcy + pred_bwbh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box
    
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

        # ---------------- Head ----------------
        cls_feat, reg_feat = self.head(feat)

        # ---------------- Pred ----------------
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)

        # ------------------- Process preds -------------------
        ## Generate anchors
        B, _, H, W = cls_feat.size()
        fmp_size = [H, W]
        anchors = self.get_anchors(fmp_size)   # [M, 4]
        anchors = anchors.to(cls_feat.device)

        ## Reshape: [B, C, H, W] -> [B, M, C], M=HW
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
        box_pred = self.decode_boxes(reg_pred, anchors)
                
        # ---------------- PostProcess ----------------
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
            cls_feat, reg_feat = self.head(feat)

            # ---------------- Pred ----------------
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)

            # ------------------- Process preds -------------------
            ## Generate anchors
            B, _, H, W = cls_feat.size()
            fmp_size = [H, W]
            anchors = self.get_anchors(fmp_size)   # [M, 4]
            anchors = anchors.to(cls_feat.device)

            ## Reshape: [B, C, H, W] -> [B, M, C], M=HW
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            box_pred = self.decode_boxes(reg_pred, anchors)
            ## Adjust mask
            if mask is not None:
                # [B, H, W]
                mask = torch.nn.functional.interpolate(mask[None].float(), size=[H, W]).bool()[0]
                # [B, H, W] -> [B, M]
                mask = mask.flatten(1)     
                
            outputs = {"pred_cls": cls_pred,   # [B, M, C]
                        "pred_reg": reg_pred,   # [B, M, 4]
                        "pred_box": box_pred,   # [B, M, 4]
                        "anchors": anchors,     # [M, 2]
                        "stride": self.stride,
                        "mask": mask}           # [B, M,]

            # ---------------- Aux Head ----------------
            if self.use_aux_head:
                aux_cls_feat, aux_reg_feat = self.aux_head(feat)
                aux_cls_pred = self.cls_pred(aux_cls_feat)
                aux_reg_pred = self.reg_pred(aux_reg_feat)

                # Reshape: [B, C, H, W] -> [B, M, C], M=HW
                aux_cls_pred = aux_cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                aux_reg_pred = aux_reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
                aux_reg_pred = self.decode_boxes(aux_reg_pred, anchors)
                    
                aux_outputs = {"pred_cls": aux_cls_pred,   # [B, M, C]
                               "pred_reg": aux_reg_pred,   # [B, M, 4]
                               "pred_box": box_pred,       # [B, M, 4]
                                "anchors": anchors,        # [M, 2]
                                "stride": self.stride,
                                "mask": mask}              # [B, M,]

                outputs['aux_outputs'] = aux_outputs

            return outputs 
