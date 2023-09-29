# ---------------------------------------------------------------------
# Copyright (c) Megvii Inc. All rights reserved.
# ---------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_ops import *
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import UniformMatcher


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss


class Criterion(nn.Module):
    """
        This code referenced to https://github.com/megvii-model/YOLOF/blob/main/playground/detection/coco/yolof/yolof_base/yolof.py
    """
    def __init__(self, cfg, device, num_classes=80):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.alpha = cfg['focal_loss_alpha']
        self.gamma = cfg['focal_loss_gamma']
        self.matcher_cfg = cfg['matcher_hpy']
        self.matcher = UniformMatcher(self.matcher_cfg['topk_candidates'])
        self.num_classes = num_classes
        self.loss_cls_weight = cfg['loss_cls_weight']
        self.loss_reg_weight = cfg['loss_reg_weight']

    def loss_labels(self, pred_cls, tgt_cls):
        """
            pred_cls: (Tensor) [N, C]
            tgt_cls:  (Tensor) [N, C]
        """
        # cls loss: [V, C]
        loss_cls = sigmoid_focal_loss(pred_cls, tgt_cls, self.alpha, self.gamma)

        return loss_cls

    def loss_bboxes(self, pred_box, tgt_box):
        """
            pred_box: (Tensor) [N, 4]
            tgt_box:  (Tensor) [N, 4]
        """
        # giou
        pred_giou = generalized_box_iou(pred_box, tgt_box)  # [N, M]
        # giou loss
        loss_reg = 1. - torch.diag(pred_giou)

        return loss_reg

    def forward(self, outputs, targets):
        """
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_box']: (Tensor) [B, M, 4]
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        # -------------------- Pre-process --------------------
        pred_box = outputs['pred_box']
        pred_cls = outputs['pred_cls'].reshape(-1, self.num_classes)
        anchor_boxes = outputs['anchors']
        masks = ~outputs['mask']
        B = len(targets)

        # -------------------- Label assignment --------------------
        indices = self.matcher(pred_box, anchor_boxes, targets)

        # [M, 4] -> [1, M, 4] -> [B, M, 4]
        anchor_boxes = box_cxcywh_to_xyxy(anchor_boxes)
        anchor_boxes = anchor_boxes[None].repeat(B, 1, 1)

        ious = []
        pos_ious = []
        for i in range(B):
            src_idx, tgt_idx = indices[i]
            # iou between predbox and tgt box
            iou, _ = box_iou(pred_box[i, ...], (targets[i]['boxes']).clone())
            if iou.numel() == 0:
                max_iou = iou.new_full((iou.size(0),), 0)
            else:
                max_iou = iou.max(dim=1)[0]
            # iou between anchorbox and tgt box
            a_iou, _ = box_iou(anchor_boxes[i], (targets[i]['boxes']).clone())
            if a_iou.numel() == 0:
                pos_iou = a_iou.new_full((0,), 0)
            else:
                pos_iou = a_iou[src_idx, tgt_idx]
            ious.append(max_iou)
            pos_ious.append(pos_iou)

        ious = torch.cat(ious)
        ignore_idx = ious > self.matcher_cfg['ignore_threshold']
        pos_ious = torch.cat(pos_ious)
        pos_ignore_idx = pos_ious < self.matcher_cfg['iou_threshold']

        src_idx = torch.cat(
            [src + idx * anchor_boxes[0].shape[0] for idx, (src, _) in
             enumerate(indices)])
        # [BM,]
        gt_cls = torch.full(pred_cls.shape[:1],self.num_classes, dtype=torch.int64, device=self.device)
        gt_cls[ignore_idx] = -1
        tgt_cls_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        tgt_cls_o[pos_ignore_idx] = -1
        gt_cls[src_idx] = tgt_cls_o.to(self.device)
        
        foreground_idxs = (gt_cls >= 0) & (gt_cls != self.num_classes)
        num_fgs = foreground_idxs.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = torch.clamp(num_fgs / get_world_size(), min=1).item()

        # -------------------- Classification loss --------------------
        gt_cls_target = torch.zeros_like(pred_cls)
        gt_cls_target[foreground_idxs, gt_cls[foreground_idxs]] = 1
        valid_idxs = (gt_cls >= 0) & masks
        loss_cls = self.loss_labels(pred_cls[valid_idxs], gt_cls_target[valid_idxs])
        loss_cls = loss_cls.sum() / num_fgs

        # -------------------- Regression loss --------------------
        tgt_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(self.device)
        tgt_boxes = tgt_boxes[~pos_ignore_idx]
        matched_pred_box = pred_box.reshape(-1, 4)[src_idx[~pos_ignore_idx.cpu()]]
        loss_box = self.loss_bboxes(matched_pred_box, tgt_boxes)
        loss_box = loss_box.sum() / num_fgs

        # total loss
        losses = self.loss_cls_weight * loss_cls + self.loss_reg_weight * loss_box

        loss_dict = dict(
            loss_cls=loss_cls,
            loss_box=loss_box,
            losses=losses
        )

        return loss_dict


def build_criterion(cfg, device, num_classes=90):
    criterion = Criterion(cfg=cfg, device=device, num_classes=num_classes)
    return criterion

    
if __name__ == "__main__":
    pass
