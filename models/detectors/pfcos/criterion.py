import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_ops import get_ious
from utils.misc import sigmoid_focal_loss
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import AlignedSimOTA


class Criterion(nn.Module):
    def __init__(self, cfg, num_classes=80, aux_loss=False):
        super().__init__()
        # ------------- Basic parameters -------------
        self.cfg = cfg
        self.num_classes = num_classes
        # ------------- Focal loss -------------
        self.alpha = cfg['focal_loss_alpha']
        self.gamma = cfg['focal_loss_gamma']
        # ------------- Loss weight -------------
        self.weight_dict = {'loss_cls': cfg['loss_cls_weight'],
                            'loss_reg': cfg['loss_reg_weight'],
                            'loss_box': cfg['loss_box_weight']}
        # ------------- Matcher -------------
        self.matcher_cfg = cfg['matcher_hpy']
        self.matcher = AlignedSimOTA(num_classes=num_classes,
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

    def loss_delta(self, pred_reg, gt_box, anchors, num_boxes=1.0):
        # xyxy -> cxcy&bwbh
        gt_cxcy = (gt_box[..., :2] + gt_box[..., 2:]) * 0.5
        gt_bwbh = torch.clamp(gt_box[..., 2:] - gt_box[..., :2], min=1e-7)
        # encode gt box
        gt_cxcy_encode = (gt_cxcy - anchors[..., :2]) / anchors[..., 2:]
        gt_bwbh_encode = torch.log(gt_bwbh / anchors[..., 2:])
        gt_box_encode = torch.cat([gt_cxcy_encode, gt_bwbh_encode], dim=-1)
        # l1 loss
        loss_box_aux = F.l1_loss(pred_reg, gt_box_encode, reduction='none')

        return loss_box_aux.sum() / num_boxes

    def get_losses(self, outputs, targets):        
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
        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)  # [N,]
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)   # [N, 4]
            # label assignment
            assigned_result = self.matcher(anchors=anchors[..., :2],
                                           pred_cls=cls_preds[batch_idx].detach(),
                                           pred_box=box_preds[batch_idx].detach(),
                                           gt_labels=tgt_labels,
                                           gt_bboxes=tgt_bboxes
                                           )
            cls_targets.append(assigned_result['assigned_labels'])
            box_targets.append(assigned_result['assigned_bboxes'])
        cls_targets = torch.cat(cls_targets, dim=0)
        box_targets = torch.cat(box_targets, dim=0)
        
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
        ## L1 loss
        reg_preds_pos = outputs['pred_reg'].view(-1, 4)[pos_inds]
        anchors_pos = outputs['anchors'].repeat(bs, 1)[pos_inds]
        loss_box = self.loss_delta(reg_preds_pos, box_targets_pos, anchors_pos, num_fgs)

        loss_dict = dict(
                loss_cls = loss_cls,
                loss_reg = loss_reg,
                loss_box = loss_box
        )

        return loss_dict
    
    def __call__(self, outputs, targets):
        return self.get_losses(outputs, targets)
               


def build_criterion(cfg, num_classes, aux_loss=False):
    criterion = Criterion(cfg, num_classes, aux_loss)

    return criterion


if __name__ == "__main__":
    pass