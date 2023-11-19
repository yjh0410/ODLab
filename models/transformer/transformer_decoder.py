# -----------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ..basic.mlp import FFN


# ---------------------------- DETR Transformer Decoder modules ----------------------------
class DETRTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 act_type="relu"):
        super().__init__()
        # ---------------- Network parameters ----------------
        ## SelfAttention
        self.self_attn  = nn.MultiheadAttention(d_model, num_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        ## CrossAttention
        self.cross_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        ## FFN
        self.ffn = FFN(d_model, mlp_ratio, dropout, act_type)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                memory,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None
                ):
        # self-attention
        q1 = k1 = self.with_pos_embed(tgt, query_pos)
        v1 = tgt
        tgt2 = self.self_attn(q1, k1, v1, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # cross-attention
        q2 = self.with_pos_embed(tgt, query_pos)
        k2 = self.with_pos_embed(memory, pos)
        v2 = memory
        tgt2 = self.cross_attn(q2, k2, v2, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # ffn
        tgt = self.ffn(tgt)
        
        return tgt


# ---------------------------- BoxRPB Transformer Decoder modules ----------------------------
## BoxPRB Attention
class BoxRPBAttention(nn.Module):
    """
        This code referenced to https://github.com/impiga/Plain-DETR/blob/main/models/global_rpe_decomp_decoder.py
    """
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        rpb_dim=512,
        feature_stride=16,
        reparam=False,
    ):
        super().__init__()
        # ----------- Basic parameters -----------
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.feature_stride = feature_stride
        self.reparam = reparam

        # ----------- Network parameters -----------
        ## QKV input proj
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        ## CPB MLP
        self.cpb_mlp1 = self.build_cpb_mlp(2, rpb_dim, num_heads)
        self.cpb_mlp2 = self.build_cpb_mlp(2, rpb_dim, num_heads)
        ## Output proj
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def build_cpb_mlp(self, in_dim, hidden_dim, out_dim):
        cpb_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, out_dim, bias=False))
        return cpb_mlp

    def forward(self,
                query,
                reference_points,
                k_input_flatten,
                v_input_flatten,
                input_spatial_shapes,
                input_padding_mask=None,
                ):
        """
            query: (torch.Tensor) [Nq, B, C]
            reference_points: (torch.Tensor) [Nq, B, C]
            k_input_flatten:  (torch.Tensor) [N, B, C]
            v_input_flatten:  (torch.Tensor) [N, B, C]
            input_spatial_shapes: (List[int, int]) [h, w]
            input_padding_mask
        """
        h, w = input_spatial_shapes
        stride = self.feature_stride
        device = query.device

        # [N, B, C] -> [B, N, C]
        query = query.permute(1, 0, 2)
        k_input_flatten = k_input_flatten.permute(1, 0, 2)
        v_input_flatten = v_input_flatten.permute(1, 0, 2)

        # ref_points format: x1y1x2y2, shape = [Nq, B, 1, 4]
        ref_pts = torch.cat([
            reference_points[:, :, :, :2] - reference_points[:, :, :, 2:] / 2,
            reference_points[:, :, :, :2] + reference_points[:, :, :, 2:] / 2,
        ], dim=-1)
        if not self.reparam:
            ref_pts[..., 0::2] *= (w * stride)
            ref_pts[..., 1::2] *= (h * stride)
        # anchor ctr points
        pos_x = torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device)[None, None, :, None] * stride  # 1, 1, w, 1
        pos_y = torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device)[None, None, :, None] * stride  # 1, 1, h, 1
        # deltas
        delta_x = ref_pts[..., 0::2] - pos_x  # nQ, B, w, 2
        delta_y = ref_pts[..., 1::2] - pos_y  # nQ, B, h, 2

        rpe_x, rpe_y = self.cpb_mlp1(delta_x), self.cpb_mlp2(delta_y)  # nQ, B, w/h, nheads
        rpe = (rpe_x[:, :, None] + rpe_y[:, :, :, None]).flatten(2, 3) # nQ, B, h, w, nheads -> nQ, B, h*w, nheads
        rpe = rpe.permute(1, 3, 0, 2) # nQ, B, h*w, nheads -> B, nheads, nQ, h*w

        # QKV input proj
        B_, N, C = k_input_flatten.shape
        k = self.k(k_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        B_, N, C = query.shape
        q = self.q(query).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        # BoxRPM Cross attention
        attn = q @ k.transpose(-2, -1)
        attn += rpe
        if input_padding_mask is not None:
            attn += input_padding_mask[:, None, None] * -100
        fmin, fmax = torch.finfo(attn.dtype).min, torch.finfo(attn.dtype).max
        torch.clip_(attn, min=fmin, max=fmax)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v
        # Output ptoj
        x = x.permute(2, 0, 1, 3).reshape(N, B_, C) # B, nheads, nQ, C -> nQ, B, nheads, C -> nQ, B, C
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

## BoxPRB Transformer decoder layer
class BoxRPBTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model   :int   = 256,
                 num_head  :int   = 8,
                 mlp_ratio :float = 4.0,
                 dropout   :float = 0.1,
                 act_type  :str   = "relu",
                 ctn_type  :str   = None,
                 rpb_dim   :int   = 512,
                 stride    :int   = 16,
                 reparam   :bool  = False):
        super().__init__()
        # ---------------- Basic parameters ----------------
        self.ctn_type = ctn_type  # cross attn type
        # ---------------- Network parameters ----------------
        ## SelfAttention
        self.self_attn  = nn.MultiheadAttention(d_model, num_head, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        ## CrossAttention
        if ctn_type == "box_rpb":
            self.cross_attn = BoxRPBAttention(d_model, num_head, rpb_dim=rpb_dim, feature_stride=stride, reparam=reparam)
        else:
            self.cross_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        ## FFN
        self.ffn = FFN(d_model, mlp_ratio, dropout, act_type)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                memory,
                tgt_mask                = None,
                memory_mask             = None,
                tgt_key_padding_mask    = None,
                memory_key_padding_mask = None,
                pos                     = None,
                query_pos               = None,
                reference_points        = None,
                memory_spatial_shapes   = None,
                ):
        # self-attention
        q1 = k1 = self.with_pos_embed(tgt, query_pos)
        v1 = tgt
        tgt2 = self.self_attn(q1, k1, v1, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # cross-attention
        if self.ctn_type == 'box_rpb':
            assert reference_points is not None
            assert memory_spatial_shapes is not None
            tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),  # Q
                                   reference_points,
                                   self.with_pos_embed(memory, pos),     # K
                                   memory,                               # V
                                   memory_spatial_shapes,
                                   memory_mask)
        else:
            tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),  # Q
                                   self.with_pos_embed(memory, pos),     # K
                                   memory,                               # V
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # ffn
        tgt = self.ffn(tgt)
        
        return tgt
