import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_ops import box_iou
from utils.misc import sigmoid_focal_loss


# Aligned Simple OTA Assigner
class AlignedSimOTA(object):
    """
        This code referenced to https://github.com/open-mmlab/mmyolo/models/task_modules/assigners/batch_dsl_assigner.py
    """
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


class AlignedOTA(object):
    """
        This code referenced to https://github.com/open-mmlab/mmyolo/models/task_modules/assigners/batch_dsl_assigner.py
    """
    def __init__(self, num_classes=80, topk_candidate=1, alpha=0.25, gamma=2.0):
        self.num_classes = num_classes
        self.topk_candidate = topk_candidate
        self.alpha = alpha
        self.gamma = gamma
        self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=50)

    def get_deltas(self, anchors, bboxes):
        assert isinstance(anchors, torch.Tensor), type(anchors)
        assert isinstance(anchors, torch.Tensor), type(anchors)

        deltas = torch.cat((anchors - bboxes[..., :2], bboxes[..., 2:] - anchors), dim=-1)

        return deltas


    @torch.no_grad()
    def __call__(self, anchors, pred_cls, pred_box, gt_labels, gt_bboxes):
        num_gt = len(gt_labels)
        num_anchors = pred_box.shape[0]

        # check gt
        if num_gt == 0 or gt_bboxes.max().item() == 0.:
            return {
                'assigned_labels': gt_labels.new_full(pred_cls[..., 0].shape,
                                                      self.num_classes,
                                                      dtype=torch.long),
                'assigned_bboxes': gt_bboxes.new_full(pred_box.shape, 0),
                'assign_metrics': gt_bboxes.new_full(pred_cls[..., 0].shape, 0)
            }
        
        # [N, M, 4], N is the number of targets, M is the number of all anchors
        deltas = self.get_deltas(anchors, gt_bboxes.unsqueeze(1))
        # [N, M]
        is_in_bboxes = deltas.min(dim=-1).values > 0.01

        # ----------------------------------- Classification cost -----------------------------------
        gt_labels_one_hot = F.one_hot(gt_labels, self.num_classes).float()
        pair_wise_cls_loss = sigmoid_focal_loss(
            pred_cls.unsqueeze(0).repeat(num_gt, 1, 1),                # [N, M, C]
            gt_labels_one_hot.unsqueeze(1).repeat(1, num_anchors, 1),  # [N, M, C]
            self.alpha,
            self.gamma
            ).sum(dim=-1) # [N, M]
        pair_wise_cls_loss_bg = sigmoid_focal_loss(
            pred_cls,
            torch.zeros_like(pred_cls),
        ).sum(dim=-1) # [M, C] -> [M]

        # ----------------------------------- Regression cost -----------------------------------
        pair_wise_ious, _ = box_iou(gt_bboxes, pred_box)  # [N, M]
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        ## foreground cost matrix
        cost_matrix = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 1e9 * (1 - is_in_bboxes.float())

        # ----------------------------------- Dynamic label assignment -----------------------------------
        # Performing Dynamic k Estimation, top_candidates = 20
        topk_ious, _ = torch.topk(pair_wise_ious * is_in_bboxes.float(), self.topk_candidate, dim=1)
        mu = pair_wise_ious.new_ones(num_gt + 1)
        mu[:-1] = torch.clamp(topk_ious.sum(1).int(), min=1).float()
        mu[-1] = num_anchors - mu[:-1].sum()
        nu = pair_wise_ious.new_ones(num_anchors)
        cost_matrix = torch.cat([cost_matrix, pair_wise_cls_loss_bg.unsqueeze(0)], dim=0)

        # Solving Optimal-Transportation-Plan pi via Sinkhorn-Iteration.
        _, pi = self.sinkhorn(mu, nu, cost_matrix)

        # Rescale pi so that the max pi for each gt equals to 1.
        rescale_factor, _ = pi.max(dim=1)
        pi = pi / rescale_factor.unsqueeze(1)

        # matched_gt_inds: [M,]
        _, matched_gt_inds = torch.max(pi, dim=0)

        # fg_mask: [M,]
        fg_mask = matched_gt_inds != num_gt

        # [M,]
        cls_target = gt_labels.new_ones(num_anchors) * self.num_classes
        cls_target[fg_mask] = gt_labels[matched_gt_inds[fg_mask]]
        print(cls_target.device, fg_mask.device, matched_gt_inds.device)

        # [M, 4]
        box_target = gt_bboxes.new_zeros((num_anchors, 4))
        gt_bboxes_ = gt_bboxes.unsqueeze(1).repeat(1, num_anchors, 1)
        print(matched_gt_inds.device, fg_mask.device, box_target.device, gt_bboxes_.device)
        box_target[fg_mask] = gt_bboxes_[matched_gt_inds[fg_mask], torch.arange(num_anchors)[fg_mask]]

        # [M,]
        iou_target = pair_wise_ious.new_zeros((num_anchors, 1))
        iou_target[fg_mask] = pair_wise_ious[matched_gt_inds[fg_mask], torch.arange(num_anchors)[fg_mask]].unsqueeze(1)
        
        return cls_target, box_target, iou_target

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
    

class SinkhornDistance(torch.nn.Module):
    r"""
        Given two empirical measures each with :math:`P_1` locations
        :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
        outputs an approximation of the regularized OT cost for point clouds.
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'none'
        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=1e-3, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v = self.eps * \
                (torch.log(
                    nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            u = self.eps * \
                (torch.log(
                    mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(
            self.M(C, U, V)).detach()
        # Sinkhorn distance
        cost = torch.sum(
            pi * C, dim=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
    