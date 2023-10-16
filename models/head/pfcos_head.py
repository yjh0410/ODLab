import torch
import torch.nn as nn
from ..basic.conv import Conv


class SingleLayerPlainFCOSHead(nn.Module):
    def __init__(self, in_dim, out_dim, num_cls_head=1, num_reg_head=1, act_type='relu', norm_type='BN'):
        super().__init__()
        self.fmp_size = None
        # ------------------ Basic parameters -------------------
        self.in_dim = in_dim
        self.num_cls_head = num_cls_head
        self.num_reg_head = num_reg_head
        self.act_type = act_type
        self.norm_type = norm_type

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
                
        # init bias
        self._init_layers()

    def _init_layers(self):
        for module in [self.cls_heads, self.reg_heads]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)
        
    def forward(self, cls_feat, reg_feat):
        cls_feat = self.cls_heads(cls_feat)
        reg_feat = self.reg_heads(reg_feat)

        return cls_feat, reg_feat

class PlainFCOSHead(nn.Module):
    def __init__(self, cfg, in_dim, out_dim, num_classes, act_type='relu', norm_type='BN'):
        super().__init__()
        self.fmp_size = None
        # ------------------ Basic parameters -------------------
        self.cfg = cfg
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.num_head = cfg['num_head']
        self.num_cls_head = cfg['num_cls_head']
        self.num_reg_head = cfg['num_reg_head']
        self.act_type = act_type
        self.norm_type = norm_type
        self.stride = cfg['out_stride']

        # ------------------ Network parameters -------------------
        ## Detection head
        self.heads = nn.ModuleList(
            SingleLayerPlainFCOSHead(in_dim, out_dim, self.num_cls_head, self.num_reg_head, self.act_type, self.norm_type)
            for _ in range(self.num_head)
        )
        ## Pred layers
        self.cls_preds = nn.ModuleList(
            nn.Conv2d(out_dim, num_classes, kernel_size=3, padding=1)
            for _ in range(self.num_head)
        )
        self.reg_preds = nn.ModuleList(
            nn.Conv2d(out_dim, 4, kernel_size=3, padding=1)
            for _ in range(self.num_head)
        )
                
        # init bias
        self._init_layers()

    def _init_layers(self):
        for module in [self.cls_preds, self.reg_preds]:
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
        for cls_pred in self.cls_preds:
            torch.nn.init.constant_(cls_pred.bias, bias_value)
        
    def get_reference_points(self, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        ref_points_y, ref_points_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        ref_points_xy = torch.stack([ref_points_x, ref_points_y], dim=-1).float().view(-1, 2) + 0.5
        ref_points_wh = torch.ones_like(ref_points_xy)
        ref_points = torch.cat([ref_points_xy * self.stride, ref_points_wh * self.stride], dim=-1)

        return ref_points
        
    def decode_boxes(self, pred_deltas, ref_points):
        """
            Input:
                ref_points:  (Tensor) -> [1, M, 2] or [M, 2]
                pred_deltas: (Tensor) -> [B, M, 4] or [M, 4]
            Output:
                pred_box: (Tensor) -> [B, M, 4] or [M, 4]
        """
        pred_cxcy = ref_points[..., :2] + pred_deltas[..., :2] * self.stride
        pred_bwbh = ref_points[..., 2:] * pred_deltas[..., 2:].exp()
        pred_box = torch.cat([pred_cxcy, pred_bwbh], dim=-1)

        return pred_box
    
    def forward(self, x, mask=None):
        # Initialize reference points
        B, _, H, W = x.size()
        ref_points = self.get_reference_points([H, W]).to(x.device)   # [M, 4]
        ref_points = ref_points[None].repeat(B, 1, 1)                 # [B, M, 4]

        # --------------------- Main process ---------------------
        output_classes = []
        output_deltas = []
        output_coords = []
        all_ref_points = [ref_points]
        cls_feat = x
        reg_feat = x
        for lid in range(len(self.heads)):
            # ------------------- Decoupled head -------------------
            cls_feat, reg_feat = self.heads[lid](cls_feat, reg_feat)

            # ------------------- Predict -------------------
            output_class = self.cls_preds[lid](cls_feat)
            output_delta = self.reg_preds[lid](reg_feat)

            # ------------------- Process preds -------------------
            ## [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
            output_class = output_class.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            output_delta = output_delta.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            output_coord = self.decode_boxes(output_delta, ref_points)

            # Iterative update reference points
            ref_points= output_coord.detach()

            output_classes.append(output_class)
            output_deltas.append(output_delta)
            output_coords.append(output_coord)
            all_ref_points.append(ref_points)

        return torch.stack(output_classes), torch.stack(output_coords), torch.stack(output_deltas), torch.stack(all_ref_points[:-1])
