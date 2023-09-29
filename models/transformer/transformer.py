# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import torch
import torch.nn as nn

from .transformer_encoder import TransformerEncoderLayer, TransformerEncoder
from .transformer_decoder import TransformerDecoderLayer, TransformerDecoder
from ..basic.mlp import MLP


class Transformer(nn.Module):
    def __init__(self,
                 d_model             :int   = 512,
                 num_encoder         :int   = 6,
                 encoder_num_head    :int   = 8,
                 encoder_mlp_ratio   :float = 4.0,
                 encoder_dropout     :float = 0.1,
                 encoder_act_type    :str   = "relu",
                 num_decoder         :int   = 6,
                 decoder_num_head    :int   = 8,
                 decoder_mlp_ratio   :float = 4.0,
                 decoder_dropout     :float = 0.1,
                 decoder_act_type    :str   = "relu",
                 num_classes         :int   = 80,
                 num_queries         :int   = 100,
                 norm_before         :bool  = False,
                 return_intermediate :bool  = False):
        super().__init__()
        # --------------- Basic parameters ---------------
        self.d_model = d_model
        self.return_intermediate = return_intermediate
        # --------------- Network parameters ---------------
        ## Transformer Encoder
        encoder_norm = nn.LayerNorm(d_model) if norm_before else None
        encoder_layer = TransformerEncoderLayer(d_model, encoder_num_head, encoder_mlp_ratio, encoder_dropout, encoder_act_type, norm_before)
        self.encoder_layers = TransformerEncoder(encoder_layer, num_encoder, encoder_norm)
        ## Transformer Decoder
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, decoder_num_head, decoder_mlp_ratio, decoder_dropout, decoder_act_type, norm_before)
        self.decoder_layers = TransformerDecoder(decoder_layer, num_decoder, decoder_norm, return_intermediate)
        # Object Query
        self.query_embed = nn.Embedding(num_queries, d_model)
        ## Output head
        self.class_embed = nn.Linear(self.d_model, num_classes)
        self.bbox_embed  = MLP(self.d_model, self.d_model, 4, 3)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        bs, c, h, w = src.shape
        # reshape: [B, C, H, W] -> [N, B, C], N = HW
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        # Transformer encoder
        memory = self.encoder_layers(src, src_key_padding_mask=mask, pos=pos_embed)

        # Transformer decoder
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder_layers(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)

        # Output head
        output_classes = self.class_embed(h)
        output_coords = self.bbox_embed(h).sigmoid()

        return output_classes, output_coords
