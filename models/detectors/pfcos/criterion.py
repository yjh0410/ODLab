import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_ops import generalized_box_iou
from utils.misc import sigmoid_focal_loss
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import OTAMatcher


class Criterion(nn.Module):
    def __init__(self, cfg, device, num_classes=80):
        super().__init__()
        # ------------- Basic parameters -------------
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        # ------------- Focal loss -------------
        self.alpha = cfg['focal_loss_alpha']
        self.gamma = cfg['focal_loss_gamma']
        # ------------- Loss weight -------------
        self.weight_dict = {'loss_cls': cfg['loss_cls_weight'],
                            'loss_reg': cfg['loss_reg_weight']}
        # ------------- Matcher -------------
        self.matcher_cfg = cfg['matcher_hpy']
        self.matcher = OTAMatcher(num_classes = num_classes,
                                  topk_candidate=self.matcher_cfg['topk_candidate'],
                                  sinkhorn_eps=self.matcher_cfg['sinkhorn_eps'],
                                  sinkhorn_iters=self.matcher_cfg['sinkhorn_iter'])

    def loss_labels(self, pred_cls, tgt_cls, num_boxes=1.0):
        """
            pred_cls: (Tensor) [N, C]
            tgt_cls:  (Tensor) [N, C]
        """
        # cls loss: [V, C]
        loss_cls = sigmoid_focal_loss(pred_cls, tgt_cls, self.alpha, self.gamma)

        return loss_cls.sum() / num_boxes

    def loss_bboxes(self, pred_box, tgt_box, num_boxes=1.0):
        """
            pred_box: (Tensor) [Nq, 4]
            tgt_box:  (Tensor) [Nq, 4]
        """
        # GIoU loss
        pred_giou = generalized_box_iou(pred_box, tgt_box)  # [N, M]
        loss_reg = 1. - torch.diag(pred_giou)

        return loss_reg.sum() / num_boxes

    def loss_deltas(self, pred_reg, pred_sca, tgt_box, anchors, stride, num_boxes=1.0):
        """
            pred_box: (Tensor) [Nq, 4]
            pred_sca: (Tensor) [Nq, 4]
            tgt_box:  (Tensor) [Nq, 4]
        """
        tgt_box_cxcy = (tgt_box[..., :2] + tgt_box[..., 2:]) * 0.5
        tgt_box_bwbh = (tgt_box[..., 2:] - tgt_box[..., :2]) * 0.5
        tgt_box_cxcy_e = (tgt_box_cxcy - anchors) / stride
        tgt_box_bwbh_e = torch.log(tgt_box_bwbh / stride)
        tgt_box_e = torch.cat([tgt_box_cxcy_e, tgt_box_bwbh_e], dim=-1)

        pred_reg[..., 2:] *= pred_sca
        loss_box = F.l1_loss(pred_reg, tgt_box_bwbh_e, reduction='none')

        return loss_box.sum() / num_boxes

    def loss_ious(self, pred_iou, tgt_iou, num_boxes=1.0):
        """
            pred_iou: (Tensor) [Nq, 1]
            tgt_iou:  (Tensor) [Nq, 1]
        """
        # IoU-aware loss
        loss_iou = F.binary_cross_entropy_with_logits(pred_iou, tgt_iou, reduction='none')

        return loss_iou.sum() / num_boxes

    def forward(self, outputs, targets):
        cls_preds = outputs['pred_cls']
        box_preds = outputs['pred_box']
        mask = ~outputs['mask']
        anchors = outputs['anchors']
        output_stride = outputs['stride']
        device = outputs['pred_cls'].device

        # --------------------- Label assignment ---------------------
        cls_targets, box_targets, iou_targets = self.matcher(anchors = anchors,
                                                             cls_preds = cls_preds,
                                                             box_preds = box_preds,
                                                             targets = targets
                                                             )

        # Reshape: [B, M, C] -> [BM, C]
        cls_preds = cls_preds.view(-1, self.num_classes)
        box_preds = box_preds.view(-1, 4)
        masks = mask.view(-1)

        cls_targets = cls_targets.flatten().to(device)
        box_targets = box_targets.view(-1, 4).to(device)
        iou_targets = iou_targets.view(-1, 1).to(device)

        foreground_idxs = (cls_targets >= 0) & (cls_targets != self.num_classes)
        num_foreground = foreground_idxs.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        # -------------------- Classification loss --------------------
        cls_targets_one_hot = torch.zeros_like(cls_preds)
        cls_targets_one_hot[foreground_idxs, cls_targets[foreground_idxs]] = 1
        valid_idxs = (cls_targets >= 0) & masks
        loss_cls = self.loss_labels(cls_preds[valid_idxs], cls_targets_one_hot[valid_idxs], num_foreground)

        # -------------------- Regression loss --------------------
        loss_reg = self.loss_bboxes(box_preds[foreground_idxs], box_targets[foreground_idxs], num_foreground)

        loss_dict = dict(
                loss_cls = loss_cls,
                loss_reg = loss_reg,
        )

        return loss_dict


# build criterion
def build_criterion(cfg, device, num_classes=80):
    criterion = Criterion(cfg=cfg, device=device, num_classes=num_classes)

    return criterion


if __name__ == "__main__":
    pass
