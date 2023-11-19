import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou, bbox2delta


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class, cost_bbox, cost_giou, alpha=0.25, gamma=2.0, box_reparam=False):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma
        self.box_reparam = box_reparam

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        # [B, Nq, C] -> [BNq, C]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # List[B, M, C] -> [BM, C]
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # -------------------- Classification cost --------------------
        neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # -------------------- Regression cost --------------------
        ## L1 cost: [Nq, M]
        if self.box_reparam:
            out_delta = outputs["pred_deltas"].flatten(0, 1)
            out_bbox_old = outputs["pred_boxes_old"].flatten(0, 1)
            tgt_delta = bbox2delta(out_bbox_old, tgt_bbox)
            cost_bbox = torch.cdist(out_delta[:, None], tgt_delta, p=1).squeeze(1)
        else:
            cost_bbox = torch.cdist(out_bbox, box_xyxy_to_cxcywh(tgt_bbox).to(out_bbox.device), p=1)
            ## GIoU cost: Nq, M]
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), tgt_bbox.to(out_bbox.device))

        # Final cost: [B, Nq, M]
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        # Label assignment
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
