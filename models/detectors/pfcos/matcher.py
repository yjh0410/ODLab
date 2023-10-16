import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_cls"].shape[:2]
        # [B, Nq, C] -> [BNq, C]
        out_prob = outputs["pred_cls"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_box"].flatten(0, 1)

        # List[B, M, C] -> [BM, C]
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # -------------------- Classification cost --------------------
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # -------------------- Regression cost --------------------
        ## L1 cost: [Nq, M]
        cost_bbox = torch.cdist(box_xyxy_to_cxcywh(out_bbox), box_xyxy_to_cxcywh(tgt_bbox).to(out_bbox.device), p=1)
        ## GIoU cost: Nq, M]
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox.to(out_bbox.device))

        # Final cost: [B, Nq, M]
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        # Label assignment
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
