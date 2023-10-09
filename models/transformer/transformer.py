# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import math
import copy
import torch
import torch.nn as nn

from .transformer_encoder import TransformerEncoderLayer, TransformerEncoder
from .transformer_decoder import TransformerDecoderLayer, TransformerDecoder
from ..basic.mlp import MLP


# ----------------------------- DETR Transformer -----------------------------
class DETRTransformer(nn.Module):
    def __init__(self,
                 d_model             :int   = 512,
                 # Encoder
                 num_encoder         :int   = 6,
                 encoder_num_head    :int   = 8,
                 encoder_mlp_ratio   :float = 4.0,
                 encoder_dropout     :float = 0.1,
                 encoder_act_type    :str   = "relu",
                 # Decoder
                 num_decoder         :int   = 6,
                 decoder_num_head    :int   = 8,
                 decoder_mlp_ratio   :float = 4.0,
                 decoder_dropout     :float = 0.1,
                 decoder_act_type    :str   = "relu",
                 # Other
                 num_classes         :int   = 90,
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


# ----------------------------- PlainDETR Transformer -----------------------------
class PlainDETRTransformer(nn.Module):
    def __init__(self,
                 d_model             :int   = 512,
                 # Encoder
                 num_encoder         :int   = 6,
                 encoder_num_head    :int   = 8,
                 encoder_mlp_ratio   :float = 4.0,
                 encoder_dropout     :float = 0.1,
                 encoder_act_type    :str   = "relu",
                 upsample            :bool  = False,
                 # Decoder
                 num_decoder         :int   = 6,
                 decoder_num_head    :int   = 8,
                 decoder_mlp_ratio   :float = 4.0,
                 decoder_dropout     :float = 0.1,
                 decoder_act_type    :str   = "relu",
                 # Other
                 num_classes         :int   = 90,
                 num_queries         :int   = 100,
                 norm_before         :bool  = False,
                 return_intermediate :bool  = False):
        super().__init__()
        # --------------- Basic parameters ---------------
        self.d_model = d_model
        self.upsample = upsample
        self.num_classes = num_classes
        self.return_intermediate = return_intermediate
        # --------------- Network parameters ---------------
        ## Transformer Encoder
        self.encoder_layers = None
        if num_encoder > 0:
            encoder_norm = nn.LayerNorm(d_model) if norm_before else None
            encoder_layer = TransformerEncoderLayer(d_model, encoder_num_head, encoder_mlp_ratio, encoder_dropout, encoder_act_type, norm_before)
            self.encoder_layers = TransformerEncoder(encoder_layer, num_encoder, encoder_norm)
        ## Upsample layer
        self.upsample_layer = None
        if upsample:
            self.upsample_layer = nn.ConvTranspose2d(d_model, d_model, kernel_size=4, padding=1, stride=2)
        ## Transformer Decoder
        self.decoder_layers = None
        if num_decoder > 0:
            decoder_norm = nn.LayerNorm(d_model)
            decoder_layer = TransformerDecoderLayer(d_model, decoder_num_head, decoder_mlp_ratio, decoder_dropout, decoder_act_type, norm_before)
            self.decoder_layers = TransformerDecoder(decoder_layer, num_decoder, decoder_norm, return_intermediate)
        ## Adaptive pos_embed
        self.adapt_pos = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        ## Object Query
        self.refpoint_embed = nn.Embedding(num_queries, 4)
        self.query_embed = nn.Embedding(num_queries, d_model)
        ## Output head
        self.class_embed = nn.Linear(self.d_model, num_classes)
        self.bbox_embed  = MLP(self.d_model, self.d_model, 4, 3)
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for _ in range(num_decoder)])
        self.bbox_embed  = nn.ModuleList([copy.deepcopy(self.bbox_embed)  for _ in range(num_decoder)])

        self.init_weight()

    def init_weight(self):
        # init class embed bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        for class_embed in self.class_embed:
            class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        # init bbox embed bias
        for bbox_embed in self.bbox_embed:
            nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        # init weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_posembed(self, x, temperature=10000):
        num_pos_feats, hs, ws = x.shape[1]//2, x.shape[2], x.shape[3]
        # generate xy coord mat
        y_embed, x_embed = torch.meshgrid(
            [torch.arange(1, hs+1, dtype=torch.float32),
             torch.arange(1, ws+1, dtype=torch.float32)])
        y_embed = y_embed / (hs + 1e-6) * self.scale
        x_embed = x_embed / (ws + 1e-6) * self.scale
    
        # [H, W] -> [1, H, W]
        y_embed = y_embed[None, :, :].to(x.device)
        x_embed = x_embed[None, :, :].to(x.device)

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)

        pos_x = torch.div(x_embed[..., None], dim_t)
        pos_y = torch.div(y_embed[..., None], dim_t)
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        # [B, H, W, C] -> [B, C, H, W]
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos_embed        

    def pos_to_posembed(self, pos, temperature=10000):
        scale = 2 * math.pi
        num_pos_feats = self.d_model // 2
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        
        return posemb

    def inverse_sigmoid(self, x):
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=1e-5)/(1 - x).clamp(min=1e-5))

    def forward(self, src, mask):
        bs, c, h, w = src.shape

        # ------------------------ Transformer Encoder ------------------------
        ## Reshape: [B, C, H, W] -> [B, N, C], N = HW
        src = src.permute(0, 2, 3, 1).reshape(bs, -1, c)
        pos_embed = self.get_posembed(src)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(bs, -1, c)
        ## Encoder layer
        if self.encoder_layers:
            for layer_id, encoder_layer in enumerate(self.encoder_layers):
                src = encoder_layer(src, src_key_padding_mask=mask, pos_embed=pos_embed)

        ## Upsample feature
        if self.upsample_layer:
            # Reshape: [B, N, C] -> [B, C, H, W]
            src = src.permute(0, 2, 1).reshape(bs, c, h, w)
            src = self.upsample_layer(src)
            # Generate pos_embed for upsampled src
            pos_embed = self.get_posembed(src)
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(bs, -1, c)

        # ------------------------ Transformer Decoder ------------------------
        ## Reshape tgt: [Nq, C] -> [B, Nq, C]
        tgt = self.query_embed.weight[None].repeat(bs, 1, 1)
        rfp_embed = self.refpoint_embed.weight.weight[None].repeat(bs, 1, 1)
        ref_point = rfp_embed.sigmoid()
        ref_points = [ref_point]
        
        ## Decoder layer
        outputs = []
        output_classes = []
        output_coords = []
        for layer_id, decoder_layer in enumerate(self.decoder_layers):
            # Adaptive pos embed
            query_pos = self.adapt_pos(self.pos_to_posembed(ref_point))
            # Decoder
            tgt = decoder_layer(tgt, src, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_pos)
            # Iter update
            delta_unsig = self.bbox_embed[layer_id](tgt)
            outputs_unsig = delta_unsig + self.inverse_sigmoid(ref_point)
            new_ref_point = outputs_unsig.sigmoid()
            ref_point = new_ref_point.detach()

            outputs.append(tgt)
            ref_points.append(ref_point)

        # ------------------------ Detection Head ------------------------
        for lid, (ref_sig, output) in enumerate(zip(ref_points[:-1], outputs)):
            ## bbox pred
            tmp = self.bbox_embed[lid](output)
            output_coord = tmp + self.inverse_sigmoid(ref_sig)
            output_coord = output_coord.sigmoid()
            ## class pred
            output_class = self.class_embed[lid](output)

            output_classes.append(output_class)
            output_coords.append(output_coord)

        return torch.stack(output_classes), torch.stack(output_coords)
