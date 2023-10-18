import torch
import torch.nn as nn
from ..basic.conv import Conv


class PlainFCOSHead(nn.Module):
    def __init__(self, cfg, in_dim, out_dim, num_cls_head=1, num_reg_head=1, act_type='relu', norm_type='BN'):
        super().__init__()
        # ------------------ Basic parameters -------------------
        self.cfg = cfg
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
        # init head layers
        for module in [self.cls_heads, self.reg_heads]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)
        
    def forward(self, x):
        cls_feat = self.cls_heads(x)
        reg_feat = self.reg_heads(x)

        return cls_feat, reg_feat
