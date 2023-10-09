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

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# ---------------------------- Vanilla Transformer Encoder modules ----------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src,
                mask = None,
                src_key_padding_mask = None,
                pos_embed = None):
        output = src

        for layer in self.layers:
            output = layer(output,
                           src_mask = mask,
                           src_key_padding_mask = src_key_padding_mask,
                           pos = pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 num_head,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 act_type="relu"):
        super().__init__()
        # ---------------- Network parameters ----------------
        ## SelfAttention
        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        ## FFN
        self.ffn = FFN(d_model, mlp_ratio, dropout, act_type)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,
                src_mask=None,
                src_key_padding_mask=None,
                pos_embed=None):
        q = k = self.with_pos_embed(src, pos_embed)
        v = src
        # self-attention
        src2 = self.self_attn(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm(src)
        # feedforward
        src = self.ffn(src)

        return src


# ---------------------------- XXX Transformer Encoder modules ----------------------------
