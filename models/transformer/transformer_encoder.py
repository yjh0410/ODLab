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
import math
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ..basic.mlp import FFN


# ---------------------------- Basic functions ----------------------------
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# ---------------------------- Transformer Encoder layer ----------------------------
class DETRTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 act_type="relu"):
        super().__init__()
        # ---------------- Network parameters ----------------
        ## self-attn
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


# ---------------------------- Transformer Encoder ----------------------------
class DETRTransformerEncoder(nn.Module):
    def __init__(self,
                 d_model             :int   = 256,
                 num_encoder         :int   = 6,
                 encoder_num_head    :int   = 8,
                 encoder_mlp_ratio   :float = 4.0,
                 encoder_dropout     :float = 0.1,
                 encoder_act_type    :str   = "relu",
                 ):
        super().__init__()
        # --------------- Basic parameters ---------------
        self.d_model = d_model
        self.scale = 2 * math.pi
        self.num_encoder = num_encoder
        self.encoder_num_head = encoder_num_head
        self.encoder_mlp_ratio = encoder_mlp_ratio
        self.encoder_dropout = encoder_dropout
        self.encoder_act_type = encoder_act_type
        # --------------- Network parameters ---------------
        ## Transformer Encoder
        self.encoder_layers = None
        if num_encoder > 0:
            encoder_layer = DETRTransformerEncoderLayer(d_model, encoder_num_head, encoder_mlp_ratio, encoder_dropout, encoder_act_type)
            self.encoder_layers = _get_clones(encoder_layer, num_encoder)

        self.init_weight()

    # -------------- Basic functions --------------
    def init_weight(self):
        # init all layer weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # -------------- Model forward --------------
    def forward(self, src, src_mask=None, pos_embed=None):
        """
            src:       (torch.Tensor) [N, B, C]
            src_mask:  (torch.Tensor) [B, N]
            pos_embed: (torch.Tensor) [N, B, C]
        """
        ## Encoder layer
        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src, src_key_padding_mask=src_mask, pos_embed=pos_embed)

        return src
