import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_ops import get_ious, generalized_box_iou
from utils.misc import sigmoid_focal_loss
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import AlignedOTA


class OTACriterion(nn.Module):
    def __init__(self, cfg, num_classes=80):
        super().__init__()
        # ------------- Basic parameters -------------
        self.cfg = cfg
        self.num_classes = num_classes
        # ------------- Focal loss -------------
        self.alpha = cfg['focal_loss_alpha']
        self.gamma = cfg['focal_loss_gamma']
        # ------------- Loss weight -------------
        self.weight_dict = {'loss_cls': cfg['loss_cls_weight'],
                            'loss_reg': cfg['loss_reg_weight']}
        # ------------- Matcher -------------
        self.matcher_cfg = cfg['matcher_hpy']
        self.matcher = AlignedOTA(num_classes=num_classes,
                                     topk_candidate=int(self.matcher_cfg['topk_candidate']),
                                     alpha=self.alpha,
                                     gamma=self.gamma
                                     )
    
    def loss_label(self, pred_cls, tgt_cls, num_boxes=1.0):
        """
            pred_cls: (Tensor) [N, C]
            tgt_cls:  (Tensor) [N, C]
        """
        # cls loss: [V, C]
        loss_cls = sigmoid_focal_loss(pred_cls, tgt_cls, self.alpha, self.gamma)

        return loss_cls.sum() / num_boxes

    def loss_giou(self, pred_box, gt_box, num_boxes=1.0):
        # regression loss
        ious = get_ious(pred_box, gt_box, "xyxy", 'giou')
        loss_box = 1.0 - ious

        return loss_box.sum() / num_boxes

    def __call__(self, outputs, targets):
        """
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_reg']: (Tensor) [B, M, 4]
            outputs['pred_box']: (Tensor) [B, M, 4]
            outputs['stride']: (Int) output stride
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 ...]
        """
        bs = outputs['pred_cls'].shape[0]
        anchors = outputs['anchors']
        mask = ~outputs['mask'].flatten()
        device = anchors.device
        # preds: [B, M, C]
        cls_preds = outputs['pred_cls']
        box_preds = outputs['pred_box']

        # --------------- label assignment ---------------
        cls_targets = []
        box_targets = []
        iou_targets = []
        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)  # [N,]
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)   # [N, 4]
            # label assignment
            cls_target, box_target, iou_target = self.matcher(anchors=anchors[..., :2],
                                           pred_cls=cls_preds[batch_idx].detach(),
                                           pred_box=box_preds[batch_idx].detach(),
                                           gt_labels=tgt_labels,
                                           gt_bboxes=tgt_bboxes
                                           )
            cls_targets.append(cls_target)
            box_targets.append(box_target)
            iou_targets.append(iou_target)
        cls_targets = torch.cat(cls_targets, dim=0)
        box_targets = torch.cat(box_targets, dim=0)
        iou_targets = torch.cat(iou_targets, dim=0)
        
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = (cls_targets >= 0) & (cls_targets != self.num_classes)
        num_fgs = pos_inds.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = (num_fgs / get_world_size()).clamp(1.0).item()
        
        # ---------------------------- Classification loss ----------------------------
        valid_inds = mask & (cls_targets >= 0)
        cls_preds = cls_preds.view(-1, self.num_classes)
        cls_targets_one_hot = torch.zeros_like(cls_preds)
        cls_targets_one_hot[pos_inds, cls_targets[pos_inds]] = 1
        loss_cls = self.loss_label(cls_preds[valid_inds], cls_targets_one_hot[valid_inds], num_fgs)

        # ---------------------------- Regression loss ----------------------------
        ## GIoU loss
        box_preds_pos = box_preds.view(-1, 4)[pos_inds]
        box_targets_pos = box_targets[pos_inds]
        loss_reg = self.loss_giou(box_preds_pos, box_targets_pos, num_fgs)

        loss_dict = {'loss_cls': loss_cls, 'loss_reg': loss_reg}

        return loss_dict

class HybridMatcher(nn.Module):
    def __init__(self, cfg, num_classes=80):
        super().__init__()
        # ------------- Basic parameters -------------
        self.cfg = cfg
        self.num_classes = num_classes
        # ------------- Focal loss -------------
        self.alpha = cfg['focal_loss_alpha']
        self.gamma = cfg['focal_loss_gamma']
        # ------------- Loss weight -------------
        self.weight_dict = {'loss_cls': cfg['loss_cls_weight'],
                            'loss_reg': cfg['loss_reg_weight']}
        # ------------- Matcher -------------
        self.matcher_cfg = cfg['matcher_hpy']
        self.one2one_matcher = None
        self.one2many_matcher = None


def build_criterion(cfg, num_classes):
    if cfg['matcher'] == 'simota':
        criterion = OTACriterion(cfg, num_classes)
    elif cfg['matcher'] == 'hybrid_matcher':
        criterion = HybridMatcher(cfg, num_classes)

    return criterion


if __name__ == "__main__":
    pass