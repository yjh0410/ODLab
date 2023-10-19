import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_ops import generalized_box_iou
from utils.misc import sigmoid_focal_loss
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import build_matcher


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
        self.matcher = build_matcher(cfg, num_classes)

    def loss_labels(self, pred_cls, gt_cls, num_boxes=1.0):
        """
            pred_cls: (Tensor) [N, C]
            gt_cls:   (Tensor) [N, C]
        """
        # cls loss: [V, C]
        loss_cls = sigmoid_focal_loss(pred_cls, gt_cls, self.alpha, self.gamma)

        return loss_cls.sum() / num_boxes

    def loss_bboxes(self, pred_box, gt_box, num_boxes=1.0):
        """
            pred_box: (Tensor) [Nq, 4]
            gt_box:   (Tensor) [Nq, 4]
        """
        # GIoU loss
        pred_giou = generalized_box_iou(pred_box, gt_box)  # [N, M]
        loss_reg = 1. - torch.diag(pred_giou)

        return loss_reg.sum() / num_boxes

    def forward(self, outputs, targets):
        mask = ~outputs['mask']
        anchors = outputs['anchors']
        # preds: [B, M, C]
        cls_preds = outputs['pred_cls']
        box_preds = outputs['pred_box']
        bs, device = cls_preds.shape[0], cls_preds.device

        # --------------------- Label assignment ---------------------
        cls_targets = []
        box_targets = []
        iou_targets = []
        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)  # [N,]
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)   # [N, 4]
            # label assignment
            cls_target, box_target, iou_target = self.matcher(anchors   = anchors[..., :2],
                                                              pred_cls  = cls_preds[batch_idx].detach(),
                                                              pred_box  = box_preds[batch_idx].detach(),
                                                              gt_labels = tgt_labels,
                                                              gt_bboxes = tgt_bboxes
                                                              )
            cls_targets.append(cls_target)
            box_targets.append(box_target)
            iou_targets.append(iou_target)

        # List[B, M, C] -> [BM, C]
        cls_targets = torch.cat(cls_targets, dim=0)  # [BM,]
        box_targets = torch.cat(box_targets, dim=0)  # [BM, 4]
        iou_targets = torch.cat(iou_targets, dim=0)  # [BM,]

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = (cls_targets >= 0) & (cls_targets != self.num_classes)
        num_fgs = pos_inds.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = (num_fgs / get_world_size()).clamp(1.0).item()
        
        # ---------------------------- Classification loss ----------------------------
        valid_inds = mask.flatten() & (cls_targets >= 0)
        cls_preds = cls_preds.view(-1, self.num_classes)
        cls_targets_one_hot = torch.zeros_like(cls_preds)
        cls_targets_one_hot[pos_inds, cls_targets[pos_inds]] = 1
        loss_cls = self.loss_labels(cls_preds[valid_inds], cls_targets_one_hot[valid_inds], num_fgs)

        # ---------------------------- Regression loss ----------------------------
        box_preds_pos = box_preds.view(-1, 4)[pos_inds]
        box_targets_pos = box_targets[pos_inds]
        loss_reg = self.loss_bboxes(box_preds_pos, box_targets_pos, num_fgs)

        loss_dict = dict(
                loss_cls = loss_cls,
                loss_reg = loss_reg,
        )

        return loss_dict        


class HybridCriterion(nn.Module):
    def __init__(self, cfg, device, num_classes=80):
        super().__init__()
        # ------------- Criterion -------------
        cfg['matcher'] = 'hungarian'
        self.criterion1 = Criterion(cfg, device, num_classes)  # use Hungarian Matcher
        cfg['matcher'] = 'simota'
        self.criterion2 = Criterion(cfg, device, num_classes)  # use SimOTA Matcher
        # ------------- Loss weight -------------
        self.weight_dict = self.criterion1.weight_dict
        self.weight_dict.update({k + "_0": v for k, v in self.criterion2.weight_dict.items()})

    def forward(self, outputs, targets):
        # -------------------- Compute loss without aux --------------------
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        loss_dict1 = self.criterion1(outputs_without_aux, targets)

        # -------------------- Compute loss with aux --------------------
        outputs_aux = outputs['aux_outputs']
        loss_dict2 = self.criterion2(outputs_aux, targets)
        loss_dict2 = {k + "_0": v for k, v in loss_dict2.items()}

        loss_dict = {}
        loss_dict.update(loss_dict1)
        loss_dict.update(loss_dict2)

        return loss_dict


# build criterion
def build_criterion(cfg, device, num_classes=80):
    if cfg['matcher'] in ['simota', 'hungarian']:
        criterion = Criterion(cfg=cfg, device=device, num_classes=num_classes)
    elif cfg['matcher'] == 'hybrid':
        assert cfg['use_aux_head'], "set use_aux_head in config as True, if you want to use hybrid matcher."
        criterion = HybridCriterion(cfg=cfg, device=device, num_classes=num_classes)

    return criterion


if __name__ == "__main__":
    pass
