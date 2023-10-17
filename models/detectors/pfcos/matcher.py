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
    def __init__(self, num_classes=80, topk_candidate=1, alpha=0.25, gamma=2.0):
        self.num_classes = num_classes
        self.topk_candidate = topk_candidate
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def __call__(self, anchors, pred_cls, pred_box, gt_labels, gt_bboxes):
        num_gt = len(gt_labels)

        # check gt
        if num_gt == 0 or gt_bboxes.max().item() == 0.:
            return {
                'assigned_labels': gt_labels.new_full(pred_cls[..., 0].shape,
                                                      self.num_classes,
                                                      dtype=torch.long),
                'assigned_bboxes': gt_bboxes.new_full(pred_box.shape, 0),
                'assign_metrics': gt_bboxes.new_full(pred_cls[..., 0].shape, 0)
            }
        
        # get inside points: [N, M]
        is_in_gt = self.find_inside_points(gt_bboxes, anchors)
        valid_mask = is_in_gt.sum(dim=0) > 0  # [M,]

        # ----------------------------------- Regression cost -----------------------------------
        pair_wise_ious, _ = box_iou(gt_bboxes, pred_box)  # [N, M]
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # ----------------------------------- Classification cost -----------------------------------
        pairwise_pred_scores = pred_cls.permute(1, 0)  # [M, C] -> [C, M]
        pairwise_pred_scores = pairwise_pred_scores[gt_labels.long(), :].float()   # [N, M]
        pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
            pairwise_pred_scores, pair_wise_ious, reduction="none") # [N, M]
        del pairwise_pred_scores

        ## foreground cost matrix
        cost_matrix = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss
        max_pad_value = torch.ones_like(cost_matrix) * 1e9
        cost_matrix = torch.where(valid_mask[None].repeat(num_gt, 1),   # [N, M]
                                  cost_matrix, max_pad_value)

        # ----------------------------------- Dynamic label assignment -----------------------------------
        (
            matched_pred_ious,
            matched_gt_inds,
            fg_mask_inboxes
        ) = self.dynamic_k_matching(
            cost_matrix,
            pair_wise_ious,
            num_gt
            )
        del pair_wise_cls_loss, cost_matrix, pair_wise_ious, pair_wise_ious_loss

        # -----------------------------------process assigned labels -----------------------------------
        assigned_labels = gt_labels.new_full(pred_cls[..., 0].shape,
                                             self.num_classes)  # [M,]
        assigned_labels[fg_mask_inboxes] = gt_labels[matched_gt_inds].squeeze(-1)
        assigned_labels = assigned_labels.long()  # [M,]

        assigned_bboxes = gt_bboxes.new_full(pred_box.shape, 0)        # [M, 4]
        assigned_bboxes[fg_mask_inboxes] = gt_bboxes[matched_gt_inds]  # [M, 4]

        assign_metrics = gt_bboxes.new_full(pred_cls[..., 0].shape, 0) # [M, 4]
        assign_metrics[fg_mask_inboxes] = matched_pred_ious            # [M, 4]

        assigned_dict = dict(
            assigned_labels=assigned_labels,
            assigned_bboxes=assigned_bboxes,
            assign_metrics=assign_metrics
            )
        
        return assigned_dict

    def find_inside_points(self, gt_bboxes, anchors):
        """
            gt_bboxes: Tensor -> [N, 2]
            anchors:   Tensor -> [M, 2]
        """
        num_anchors = anchors.shape[0]
        num_gt = gt_bboxes.shape[0]

        anchors_expand = anchors.unsqueeze(0).repeat(num_gt, 1, 1)           # [N, M, 2]
        gt_bboxes_expand = gt_bboxes.unsqueeze(1).repeat(1, num_anchors, 1)  # [N, M, 4]

        # offset
        lt = anchors_expand - gt_bboxes_expand[..., :2]
        rb = gt_bboxes_expand[..., 2:] - anchors_expand
        bbox_deltas = torch.cat([lt, rb], dim=-1)

        is_in_gts = bbox_deltas.min(dim=-1).values > 0

        return is_in_gts
    
    def dynamic_k_matching(self, cost_matrix, pairwise_ious, num_gt):
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets.

        Args:
            cost_matrix (Tensor): Cost matrix.
            pairwise_ious (Tensor): Pairwise iou matrix.
            num_gt (int): Number of gt.
            valid_mask (Tensor): Mask for valid bboxes.
        Returns:
            tuple: matched ious and gt indexes.
        """
        matching_matrix = torch.zeros_like(cost_matrix, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk_candidate, pairwise_ious.size(1))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=1)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        # sorting the batch cost matirx is faster than topk
        _, sorted_indices = torch.sort(cost_matrix, dim=1)
        for gt_idx in range(num_gt):
            topk_ids = sorted_indices[gt_idx, :dynamic_ks[gt_idx]]
            matching_matrix[gt_idx, :][topk_ids] = 1

        del topk_ious, dynamic_ks, topk_ids

        prior_match_gt_mask = matching_matrix.sum(0) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost_matrix[:, prior_match_gt_mask], dim=0)
            matching_matrix[:, prior_match_gt_mask] *= 0
            matching_matrix[cost_argmin, prior_match_gt_mask] = 1

        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(0)[fg_mask_inboxes]
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

        return matched_pred_ious, matched_gt_inds, fg_mask_inboxes