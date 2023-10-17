import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_ops import get_ious, generalized_box_iou
from utils.misc import sigmoid_focal_loss
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import AlignedSimOTA, HungarianMatcher


class HungarianCriterion(nn.Module):
    def __init__(self, cfg, num_classes=80):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        # ------------ Basic parameters ------------
        self.cfg = cfg
        self.num_classes = num_classes
        self.losses = ['labels', 'boxes']
        # ------------- Focal loss -------------
        self.alpha = cfg['focal_loss_alpha']
        self.gamma = cfg['focal_loss_gamma']
        # ------------ Matcher ------------
        self.matcher_cfg = cfg['matcher_hpy']
        self.matcher = HungarianMatcher(cost_class = cfg['matcher_hpy']['cost_cls_weight'],
                                        cost_bbox  = cfg['matcher_hpy']['cost_box_weight'],
                                        cost_giou  = cfg['matcher_hpy']['cost_giou_weight'],
                                        alpha      = self.alpha,
                                        gamma      = self.gamma)
        # ------------- Loss weight -------------
        self.weight_dict = {'loss_cls': cfg['loss_cls_weight'],
                            'loss_reg': cfg['loss_reg_weight']}

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

    def loss_labels(self, pred_cls, targets, indices, mask, num_boxes):
        # prepare assignment indexs # class targets
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(pred_cls.device)
        target_classes = torch.full(pred_cls.shape[:2], self.num_classes, dtype=torch.int64, device=pred_cls.device)
        target_classes[idx] = target_classes_o

        # to one-hot labels
        cls_targets = torch.zeros([*pred_cls.shape[:2], pred_cls.shape[2] + 1], dtype=pred_cls.dtype, layout=pred_cls.layout)
        cls_targets = cls_targets.to(pred_cls.device)
        cls_targets.scatter_(2, target_classes.unsqueeze(-1), 1)
        cls_targets = cls_targets[:, :, :-1]

        # compute focal loss
        valid_indices = mask.flatten() > 0
        pred_cls = pred_cls.view(-1, self.num_classes)[valid_indices]
        cls_targets = cls_targets.view(-1, self.num_classes)[valid_indices]
        loss_cls = sigmoid_focal_loss(pred_cls, cls_targets, self.alpha, self.gamma)

        return loss_cls.sum() / num_boxes

    def loss_boxes(self, pred_box, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # prepare assignment indexs # bbox targets
        idx = self._get_src_permutation_idx(indices)
        pred_box = pred_box[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(pred_box.device)
        # comput giou loss
        bbox_giou = generalized_box_iou((pred_box), target_boxes)
        loss_giou = 1 - torch.diag(bbox_giou)
        
        return loss_giou.sum() / num_boxes

    def forward(self, outputs, targets):
        pred_cls = outputs['pred_cls']
        pred_box = outputs['pred_box']
        mask = ~outputs['mask']

        # -------------------- Label assignment --------------------
        indices = self.matcher(pred_cls, pred_box, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # -------------------- Classification loss --------------------
        loss_cls = self.loss_labels(pred_cls, targets, indices, mask, num_boxes)

        # -------------------- Regression loss --------------------
        loss_reg = self.loss_boxes(pred_box, targets, indices, num_boxes)
        
        loss_dict = {'loss_cls': loss_cls, 'loss_reg': loss_reg}

        return loss_dict

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
    if cfg['matcher'] == 'sim_ota':
        criterion = OTACriterion(cfg, num_classes)
    elif cfg['matcher'] == 'hungarian':
        criterion = HungarianCriterion(cfg, num_classes)
    elif cfg['matcher'] == 'hybrid_matcher':
        criterion = HybridMatcher(cfg, num_classes)

    return criterion


if __name__ == "__main__":
    pass