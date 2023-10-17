import torch
import torch.nn.functional as F
from utils.misc import sigmoid_focal_loss, SinkhornDistance


@torch.no_grad()
def get_ious_and_iou_loss(inputs,
                          targets,
                          weight=None,
                          box_mode="xyxy",
                          loss_type="iou",
                          reduction="none"):
    """
    Compute iou loss of type ['iou', 'giou', 'linear_iou']

    Args:
        inputs (tensor): pred values
        targets (tensor): target values
        weight (tensor): loss weight
        box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
        loss_type (str): 'giou' or 'iou' or 'linear_iou'
        reduction (str): reduction manner

    Returns:
        loss (tensor): computed iou loss.
    """
    if box_mode == "ltrb":
        inputs = torch.cat((-inputs[..., :2], inputs[..., 2:]), dim=-1)
        targets = torch.cat((-targets[..., :2], targets[..., 2:]), dim=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    eps = torch.finfo(torch.float32).eps

    inputs_area = (inputs[..., 2] - inputs[..., 0]).clamp_(min=0) \
        * (inputs[..., 3] - inputs[..., 1]).clamp_(min=0)
    targets_area = (targets[..., 2] - targets[..., 0]).clamp_(min=0) \
        * (targets[..., 3] - targets[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(inputs[..., 2], targets[..., 2])
                   - torch.max(inputs[..., 0], targets[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(inputs[..., 3], targets[..., 3])
                   - torch.max(inputs[..., 1], targets[..., 1])).clamp_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = targets_area + inputs_area - area_intersect
    ious = area_intersect / area_union.clamp(min=eps)

    if loss_type == "iou":
        loss = -ious.clamp(min=eps).log()
    elif loss_type == "linear_iou":
        loss = 1 - ious
    elif loss_type == "giou":
        g_w_intersect = torch.max(inputs[..., 2], targets[..., 2]) \
            - torch.min(inputs[..., 0], targets[..., 0])
        g_h_intersect = torch.max(inputs[..., 3], targets[..., 3]) \
            - torch.min(inputs[..., 1], targets[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1 - gious
    else:
        raise NotImplementedError
    if weight is not None:
        loss = loss * weight.view(loss.size())
        if reduction == "mean":
            loss = loss.sum() / max(weight.sum().item(), eps)
    else:
        if reduction == "mean":
            loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()

    return ious, loss


class OTAMatcher(object):
    def __init__(self, num_classes, topk_candidate=4, center_sampling_radius=1.5, sinkhorn_eps=0.1, sinkhorn_iters=50):
        self.num_classes = num_classes
        self.topk_candidate = topk_candidate
        self.center_sampling_radius = center_sampling_radius
        self.sinkhorn = SinkhornDistance(sinkhorn_eps, sinkhorn_iters)

    def get_deltas(self, anchors, bboxes):
        assert isinstance(anchors, torch.Tensor), type(anchors)
        assert isinstance(anchors, torch.Tensor), type(anchors)

        deltas = torch.cat((anchors - bboxes[..., :2], bboxes[..., 2:] - anchors), dim=-1)

        return deltas

    @torch.no_grad()
    def __call__(self, stride, anchors, cls_preds, reg_preds, box_preds, targets):
        cls_targets = []
        box_targets = []
        iou_targets = []
        assigned_units = []
        device = anchors.device

        # --------------------- Perform label assignment on each image ---------------------
        for target, cls_pred, reg_pred, box_pred in zip(targets, cls_preds, reg_preds, box_preds):
            gt_labels = target["labels"].to(device)
            gt_bboxes = target["boxes"].to(device)

            # [N, M, 4], N is the number of targets, M is the number of all anchors
            deltas = self.get_deltas(anchors, gt_bboxes.unsqueeze(1))
            is_in_bboxes = deltas.min(dim=-1).values > 0.01

            # targets bbox centers: [N, 2]
            centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) * 0.5
            is_in_centers = []
            radius = stride * self.center_sampling_radius
            center_bboxes = torch.cat((
                torch.max(centers - radius, gt_bboxes[:, :2]),
                torch.min(centers + radius, gt_bboxes[:, 2:]),
            ), dim=-1)
            # [N, Mi, 2]
            center_deltas = self.get_deltas(anchors, center_bboxes.unsqueeze(1))
            is_in_centers.append(center_deltas.min(dim=-1).values > 0)
            # [N, M], M = M1 + M2 + ... + MF
            is_in_centers = torch.cat(is_in_centers, dim=1)

            del centers, center_bboxes, deltas, center_deltas

            # [N, M]
            is_in_bboxes = (is_in_bboxes & is_in_centers)

            num_gt = len(gt_labels)   # N
            num_anchor = anchors.shape[0]         # M
            shape = (num_gt, num_anchor, -1)      # [N, M, -1]

            with torch.no_grad():
                # -------------------- Classification cost --------------------
                ## Foreground cls cost
                gt_labels_ot = F.one_hot(gt_labels, self.num_classes).float()
                pair_wise_cls_pred  = cls_pred.unsqueeze(0).expand(shape)      # [M, C] -> [1, M, C] -> [N, M, C]
                pair_wise_cls_label = gt_labels_ot.unsqueeze(1).expand(shape)  # [N, C] -> [N, 1, C] -> [N, M, C]
                cost_cls = sigmoid_focal_loss(pair_wise_cls_pred, pair_wise_cls_label)
                cost_cls = cost_cls.sum(dim=-1) # [N, M, C] -> [N, M]
                ## Background cls cost
                cost_cls_bg = sigmoid_focal_loss(cls_pred, torch.zeros_like(cls_pred))
                cost_cls_bg = cost_cls_bg.sum(dim=-1) # [M, C] -> [M]

                # -------------------- Regression cost --------------------
                pair_wise_box_pred  = box_pred.unsqueeze(0).expand(shape)
                pair_wise_box_label = gt_bboxes.unsqueeze(1).expand(shape)
                ## [N, M]
                pair_wise_ious, cost_reg = get_ious_and_iou_loss(
                    pair_wise_box_pred, pair_wise_box_label, box_mode='xyxy', loss_type='iou')

                # Fully cost matrix
                cost = cost_cls + 3.0 * cost_reg + 1e6 * (1 - is_in_bboxes.float())

                # --------------------- Dynamic assignment with SinkHorn ---------------------
                ## Prepare for Sinkhorn
                topk_ious, _ = torch.topk(pair_wise_ious * is_in_bboxes.float(), self.topk_candidate, dim=1)
                mu = pair_wise_ious.new_ones(num_gt + 1)
                mu[:-1] = torch.clamp(topk_ious.sum(1).int(), min=1).float()
                mu[-1] = num_anchor - mu[:-1].sum()
                nu = pair_wise_ious.new_ones(num_anchor)
                cost_matrix = torch.cat([cost, cost_cls_bg.unsqueeze(0)], dim=0)

                ## Run Sinkhorn
                _, pi = self.sinkhorn(mu, nu, cost_matrix)

                ## Rescale pi so that the max pi for each gt equals to 1.
                rescale_factor, _ = pi.max(dim=1)
                pi = pi / rescale_factor.unsqueeze(1)

                ## Get matched_gt_inds: [M,]
                max_assigned_units, matched_gt_inds = torch.max(pi, dim=0)

                # --------------------- Post-process assignment results ---------------------
                # foreground mask: [M,]
                fg_mask = matched_gt_inds != num_gt

                # [M,]
                gt_classes_i = gt_labels.new_ones(num_anchor) * self.num_classes
                gt_classes_i[fg_mask] = gt_labels[matched_gt_inds[fg_mask]]
                cls_targets.append(gt_classes_i)
                assigned_units.append(max_assigned_units)

                # [M, 4]
                gt_bboxes_i = gt_bboxes.new_zeros((num_anchor, 4))
                gt_bboxes_i[fg_mask] = \
                    pair_wise_box_label[matched_gt_inds[fg_mask], torch.arange(num_anchor, device=device)[fg_mask]]
                box_targets.append(gt_bboxes_i)

                # [M,]
                gt_ious_i = pair_wise_ious.new_zeros((num_anchor, 1))
                gt_ious_i[fg_mask] = \
                    pair_wise_ious[matched_gt_inds[fg_mask], torch.arange(num_anchor,  device=device)[fg_mask]].unsqueeze(1)
                iou_targets.append(gt_ious_i)

        # [B, M, C]
        cls_targets = torch.stack(cls_targets)
        box_targets = torch.stack(box_targets)
        iou_targets = torch.stack(iou_targets)

        return cls_targets, box_targets, iou_targets
    