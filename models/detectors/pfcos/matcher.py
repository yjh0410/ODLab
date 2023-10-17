import torch
import torch.nn.functional as F
from utils.box_ops import box_iou
from utils.misc import sigmoid_focal_loss, SinkhornDistance


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

            # [N, M, 4], get inside points
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
            # [N, M, 2]
            center_deltas = self.get_deltas(anchors, center_bboxes.unsqueeze(1))
            is_in_centers.append(center_deltas.min(dim=-1).values > 0)
            # [N, M], get central neighborhood points
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
                ## [N, M]
                pair_wise_ious, _ = box_iou(gt_bboxes, box_pred)  # [N, M]
                cost_reg = -torch.log(pair_wise_ious + 1e-8)

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
                pair_wise_box_label = gt_bboxes.unsqueeze(1).expand(shape)
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


class SimOTAMatcher(object):
    def __init__(self, num_classes=80, center_sampling_radius=1.5, topk_candidate=1, alpha=0.25, gamma=2.0):
        self.num_classes = num_classes
        self.topk_candidate = topk_candidate
        self.center_sampling_radius = center_sampling_radius
        self.alpha = alpha
        self.gamma = gamma


    @torch.no_grad()
    def __call__(self, stride, anchors, pred_cls, pred_box, tgt_labels, tgt_bboxes):
        num_gt = len(tgt_labels)
        num_anchor = anchors.shape[0]        

        # ----------------------- Find inside points -----------------------
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            tgt_bboxes, anchors, stride, num_anchor, num_gt)
        cls_preds = pred_cls[fg_mask].float()   # [Mp, C]
        box_preds = pred_box[fg_mask].float()   # [Mp, 4]

        # ----------------------- Reg cost -----------------------
        pair_wise_ious, _ = box_iou(tgt_bboxes, box_preds)      # [N, Mp]
        reg_cost = -torch.log(pair_wise_ious + 1e-8)            # [N, Mp]

        # ----------------------- Cls cost -----------------------
        # [Mp, C] -> [N, Mp, C]
        cls_preds_expand = cls_preds.unsqueeze(0).repeat(num_gt, 1, 1)
        # prepare cls_target
        cls_targets = F.one_hot(tgt_labels.long(), self.num_classes).float()
        cls_targets = cls_targets.unsqueeze(1).repeat(1, cls_preds_expand.size(1), 1)
        cls_targets *= pair_wise_ious.unsqueeze(-1)  # iou-aware
        # [N, Mp]
        cls_cost = F.binary_cross_entropy_with_logits(cls_preds_expand, cls_targets, reduction="none").sum(-1)
        del cls_preds_expand

        #----------------------- Dynamic K-Matching -----------------------
        cost_matrix = (
            cls_cost
            + 3.0 * reg_cost
            + 100000.0 * (~is_in_boxes_and_center)
        ) # [N, Mp]

        (
            assigned_labels,         # [num_fg,]
            assigned_ious,           # [num_fg,]
            assigned_indexs,         # [num_fg,]
        ) = self.dynamic_k_matching(
            cost_matrix,
            pair_wise_ious,
            tgt_labels,
            num_gt,
            fg_mask
            )
        del cls_cost, cost_matrix, pair_wise_ious, reg_cost

        return fg_mask, assigned_labels, assigned_ious, assigned_indexs

    def get_in_boxes_info(
        self,
        gt_bboxes,   # [N, 4]
        anchors,     # [M, 2]
        strides,     # [M,]
        num_anchors, # M
        num_gt,      # N
        ):
        # anchor center
        x_centers = anchors[:, 0]
        y_centers = anchors[:, 1]

        # [M,] -> [1, M] -> [N, M]
        x_centers = x_centers.unsqueeze(0).repeat(num_gt, 1)
        y_centers = y_centers.unsqueeze(0).repeat(num_gt, 1)

        # [N,] -> [N, 1] -> [N, M]
        gt_bboxes_l = gt_bboxes[:, 0].unsqueeze(1).repeat(1, num_anchors) # x1
        gt_bboxes_t = gt_bboxes[:, 1].unsqueeze(1).repeat(1, num_anchors) # y1
        gt_bboxes_r = gt_bboxes[:, 2].unsqueeze(1).repeat(1, num_anchors) # x2
        gt_bboxes_b = gt_bboxes[:, 3].unsqueeze(1).repeat(1, num_anchors) # y2

        b_l = x_centers - gt_bboxes_l
        b_r = gt_bboxes_r - x_centers
        b_t = y_centers - gt_bboxes_t
        b_b = gt_bboxes_b - y_centers
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center
        center_radius = self.center_sampling_radius

        # [N, 2]
        gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) * 0.5
        
        # [1, M]
        center_radius_ = center_radius * strides

        gt_bboxes_l = gt_centers[:, 0].unsqueeze(1).repeat(1, num_anchors) - center_radius_ # x1
        gt_bboxes_t = gt_centers[:, 1].unsqueeze(1).repeat(1, num_anchors) - center_radius_ # y1
        gt_bboxes_r = gt_centers[:, 0].unsqueeze(1).repeat(1, num_anchors) + center_radius_ # x2
        gt_bboxes_b = gt_centers[:, 1].unsqueeze(1).repeat(1, num_anchors) + center_radius_ # y2

        c_l = x_centers - gt_bboxes_l
        c_r = gt_bboxes_r - x_centers
        c_t = y_centers - gt_bboxes_t
        c_b = gt_bboxes_b - y_centers
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center
     
    def dynamic_k_matching(
        self, 
        cost, 
        pair_wise_ious, 
        gt_classes, 
        num_gt, 
        fg_mask
        ):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(self.topk_candidate, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        assigned_indexs = matching_matrix[:, fg_mask_inboxes].argmax(0)
        assigned_labels = gt_classes[assigned_indexs]

        assigned_ious = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return assigned_labels, assigned_ious, assigned_indexs
    