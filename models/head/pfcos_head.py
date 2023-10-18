import math
import torch
import torch.nn as nn
from ..basic.conv import Conv


class PlainFCOSHead(nn.Module):
    def __init__(self, cfg, in_dim, out_dim, num_classes, num_cls_head=1, num_reg_head=1, act_type='relu', norm_type='BN'):
        super().__init__()
        self.fmp_size = None
        self.DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)
        # ------------------ Basic parameters -------------------
        self.cfg = cfg
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.num_cls_head = num_cls_head
        self.num_reg_head = num_reg_head
        self.act_type = act_type
        self.norm_type = norm_type
        self.stride = cfg['out_stride']

        # ------------------ Network parameters -------------------
        ## cls head
        cls_heads = []
        self.cls_head_dim = out_dim
        for i in range(self.num_cls_head):
            if i == 0:
                cls_heads.append(
                    Conv(in_dim, self.cls_head_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type)
                        )
            else:
                cls_heads.append(
                    Conv(self.cls_head_dim, self.cls_head_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type)
                        )
        ## reg head
        reg_heads = []
        self.reg_head_dim = out_dim
        for i in range(self.num_reg_head):
            if i == 0:
                reg_heads.append(
                    Conv(in_dim, self.reg_head_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type)
                        )
            else:
                reg_heads.append(
                    Conv(self.reg_head_dim, self.reg_head_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type)
                        )
        self.cls_heads = nn.Sequential(*cls_heads)
        self.reg_heads = nn.Sequential(*reg_heads)

        ## pred layers
        self.cls_pred = nn.Conv2d(self.cls_head_dim, num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(self.reg_head_dim, 4, kernel_size=3, padding=1)
                
        # init bias
        self._init_layers()

    def _init_layers(self):
        for module in [self.cls_heads, self.reg_heads, self.cls_pred, self.reg_pred]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)
        # init the bias of cls pred
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        torch.nn.init.constant_(self.cls_pred.bias, bias_value)
        
    def get_anchors(self, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchors = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        anchors *= self.stride

        return anchors
        
    def decode_boxes(self, pred_reg, anchors):
        """
            pred_reg: (Tensor) [B, M, 4] or [M, 4]
            anchors:  (Tensor) [1, M, 2] or [M, 2]
        """
        pred_cxcy = anchors + pred_reg[..., :2] * self.stride
        pred_bwbh = torch.exp(pred_reg[..., 2:].clamp(max=self.DEFAULT_SCALE_CLAMP)) * self.stride

        pred_x1y1 = pred_cxcy - pred_bwbh * 0.5
        pred_x2y2 = pred_cxcy + pred_bwbh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box
    
    def forward(self, feat, mask=None):
        # ------------------- Decoupled head -------------------
        cls_feat = self.cls_heads(feat)
        reg_feat = self.reg_heads(feat)

        # ------------------- Generate anchor box -------------------
        B, _, H, W = cls_feat.size()
        fmp_size = [H, W]
        anchors = self.get_anchors(fmp_size)   # [M, 4]
        anchors = anchors.to(cls_feat.device)

        # ------------------- Predict -------------------
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)

        # ------------------- Process preds -------------------
        ## [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
        box_pred = self.decode_boxes(reg_pred, anchors)
        ## Adjust mask
        if mask is not None:
            # [B, H, W]
            mask = torch.nn.functional.interpolate(mask[None].float(), size=[H, W]).bool()[0]
            # [B, H, W] -> [B, M]
            mask = mask.flatten(1)     
            
        outputs = {"pred_cls": cls_pred,   # [B, M, C]
                   "pred_reg": reg_pred,   # [B, M, 4]
                   "pred_box": box_pred,   # [B, M, 4]
                   "anchors": anchors,     # [M, 2]
                   "stride": self.stride,
                   "mask": mask}           # [B, M,]

        return outputs 
