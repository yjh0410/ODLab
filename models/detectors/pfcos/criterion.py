import torch
import torch.nn as nn
import torch.nn.functional as F

from .matcher import HungarianMatcher
from utils.misc import sigmoid_focal_loss
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
from utils.distributed_utils import is_dist_avail_and_initialized, get_world_size


class Criterion(nn.Module):
    def __init__(self, cfg, num_classes=80):
        super().__init__()
        # ------------ Basic parameters ------------
        self.cfg = cfg
        self.num_classes = num_classes
        self.losses = ['labels', 'boxes']
        # ------------- Focal loss -------------
        self.alpha = cfg['focal_loss_alpha']
        self.gamma = cfg['focal_loss_gamma']
        # ------------ Matcher ------------
        self.matcher = HungarianMatcher(cost_class = cfg['matcher_hpy']['cost_cls_weight'],
                                        cost_bbox  = cfg['matcher_hpy']['cost_box_weight'],
                                        cost_giou  = cfg['matcher_hpy']['cost_giou_weight'])
        # ------------- Loss weight -------------
        self.weight_dict = {'loss_cls':  cfg['loss_cls_weight'],
                            'loss_box':  cfg['loss_box_weight'],
                            'loss_giou': cfg['loss_giou_weight']}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)"""
        src_logits = outputs['pred_logits']
        # prepare class targets
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64,).to(src_logits.device)
        target_classes[idx] = target_classes_o
        # get one-hot class labels
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        # compute class losses
        loss_cls = sigmoid_focal_loss(src_logits, target_classes_onehot, self.alpha, self.gamma)
        loss_cls = loss_cls.sum() / num_boxes

        return loss_cls

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        # prepare bbox targets
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_boxes.device)
        # compute L1 loss
        loss_bbox = F.l1_loss(src_boxes, box_xyxy_to_cxcywh(target_boxes), reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes
        # compute GIoU loss
        bbox_giou = generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), target_boxes)
        loss_giou = 1 - torch.diag(bbox_giou)
        loss_giou = loss_giou.sum() / num_boxes

        return loss_bbox, loss_giou

    def forward(self, outputs, targets):
        # ---------------- Label assignment ----------------
        indices = self.matcher(outputs, targets)

        # ---------------- Number of foregrounds ----------------
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # ---------------- Classification loss ----------------
        loss_cls = self.loss_labels(outputs, targets, indices, num_boxes)

        # ---------------- Regression loss ----------------
        loss_box, loss_giou = self.loss_boxes(outputs, targets, indices, num_boxes)

        loss_dict = dict(
            loss_cls = loss_cls,
            loss_box = loss_box,
            loss_giou = loss_giou
        )

        return loss_dict


# build criterion
def build_criterion(cfg, num_classes):
    criterion = Criterion(cfg, num_classes)

    return criterion
    