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


# ---------------------------- Basic functions ----------------------------
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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


# ---------------------------- PlainDETR Transformer Decoder modules ----------------------------
class PlainDETRTransformerDecoderLayer(nn.Module):
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

