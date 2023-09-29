import torch
import torch.nn.functional as F
from .matcher import FcosMatcher
from utils.box_ops import *
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized


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


class Criterion(object):
    def __init__(self, cfg, device, num_classes=80):
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.alpha = cfg['alpha']
        self.gamma = cfg['gamma']
        self.loss_cls_weight = cfg['loss_cls_weight']
        self.loss_reg_weight = cfg['loss_reg_weight']
        self.loss_ctn_weight = cfg['loss_ctn_weight']
        self.matcher_cfg = cfg['matcher_hpy']
        self.matcher = FcosMatcher(num_classes,
                                   self.matcher_cfg['center_sampling_radius'],
                                   self.matcher_cfg['object_sizes_of_interest'],
                                   [1., 1., 1., 1.]
                                   )

    def loss_labels(self, pred_cls, tgt_cls, num_boxes=1.0):
        """
            pred_cls: (Tensor) [N, C]
            tgt_cls:  (Tensor) [N, C]
        """
        # cls loss: [V, C]
        loss_cls = sigmoid_focal_loss(pred_cls, tgt_cls, self.alpha, self.gamma, reduction='none')

        return loss_cls.sum() / num_boxes

    def loss_bboxes(self, pred_delta, tgt_delta, bbox_quality=None, num_boxes=1.0):
        """
            pred_box: (Tensor) [N, 4]
            tgt_box:  (Tensor) [N, 4]
        """
        pred_delta = torch.cat((-pred_delta[..., :2], pred_delta[..., 2:]), dim=-1)
        tgt_delta = torch.cat((-tgt_delta[..., :2], tgt_delta[..., 2:]), dim=-1)

        eps = torch.finfo(torch.float32).eps

        pred_area = (pred_delta[..., 2] - pred_delta[..., 0]).clamp_(min=0) \
            * (pred_delta[..., 3] - pred_delta[..., 1]).clamp_(min=0)
        tgt_area = (tgt_delta[..., 2] - tgt_delta[..., 0]).clamp_(min=0) \
            * (tgt_delta[..., 3] - tgt_delta[..., 1]).clamp_(min=0)

        w_intersect = (torch.min(pred_delta[..., 2], tgt_delta[..., 2])
                    - torch.max(pred_delta[..., 0], tgt_delta[..., 0])).clamp_(min=0)
        h_intersect = (torch.min(pred_delta[..., 3], tgt_delta[..., 3])
                    - torch.max(pred_delta[..., 1], tgt_delta[..., 1])).clamp_(min=0)

        area_intersect = w_intersect * h_intersect
        area_union = tgt_area + pred_area - area_intersect
        ious = area_intersect / area_union.clamp(min=eps)

       # giou
        g_w_intersect = torch.max(pred_delta[..., 2], tgt_delta[..., 2]) \
            - torch.min(pred_delta[..., 0], tgt_delta[..., 0])
        g_h_intersect = torch.max(pred_delta[..., 3], tgt_delta[..., 3]) \
            - torch.min(pred_delta[..., 1], tgt_delta[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss_box = 1 - gious

        if bbox_quality is not None:
            loss_box = loss_box * bbox_quality.view(loss_box.size())

        return loss_box.sum() / num_boxes

    def __call__(self, outputs, targets):
        """
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_reg']: (Tensor) [B, M, 4]
            outputs['pred_ctn']: (Tensor) [B, M, 1]
            outputs['strides']: (List) [8, 16, 32, ...] stride of the model output
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        # -------------------- Pre-process --------------------
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']
        pred_cls = torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes)
        pred_delta = torch.cat(outputs['pred_reg'], dim=1).view(-1, 4)
        pred_ctn = torch.cat(outputs['pred_ctn'], dim=1).view(-1, 1)
        masks = torch.cat(outputs['mask'], dim=1).view(-1)

        # -------------------- Label Assignment --------------------
        gt_classes, gt_deltas, gt_centerness = self.matcher(fpn_strides, anchors, targets)
        gt_classes = gt_classes.flatten().to(device)
        gt_deltas = gt_deltas.view(-1, 4).to(device)
        gt_centerness = gt_centerness.view(-1, 1).to(device)

        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        num_foreground_centerness = gt_centerness[foreground_idxs].sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground_centerness)
        num_targets = torch.clamp(num_foreground_centerness / get_world_size(), min=1).item()

        # -------------------- classification loss --------------------
        gt_classes_target = torch.zeros_like(pred_cls)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1
        valid_idxs = (gt_classes >= 0) & masks
        loss_labels = self.loss_labels(
            pred_cls[valid_idxs], gt_classes_target[valid_idxs], num_foreground)

        # -------------------- regression loss --------------------
        loss_bboxes = self.loss_bboxes(
            pred_delta[foreground_idxs], gt_deltas[foreground_idxs], gt_centerness[foreground_idxs], num_targets)

        # -------------------- centerness loss --------------------
        loss_centerness = F.binary_cross_entropy_with_logits(
            pred_ctn[foreground_idxs],  gt_centerness[foreground_idxs], reduction='none')
        loss_centerness = loss_centerness.sum() / num_foreground

        # total loss
        losses = self.loss_cls_weight * loss_labels + \
                 self.loss_reg_weight * loss_bboxes + \
                 self.loss_ctn_weight * loss_centerness

        loss_dict = dict(
                loss_cls = loss_labels,
                loss_reg = loss_bboxes,
                loss_ctn = loss_centerness,
                losses = losses
        )

        return loss_dict
    

# build criterion
def build_criterion(cfg, device, num_classes=80):
    criterion = Criterion(cfg=cfg, device=device, num_classes=num_classes)
    return criterion


if __name__ == "__main__":
    pass
