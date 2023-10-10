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

from .transformer_encoder import DETRTransformerEncoderLayer, PlainDETRTransformerEncoderLayer
from .transformer_decoder import DETRTransformerDecoderLayer, PlainDETRTransformerDecoderLayer
from ..basic.mlp import MLP


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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
        encoder_layer = DETRTransformerEncoderLayer(d_model, encoder_num_head, encoder_mlp_ratio, encoder_dropout, encoder_act_type)
        self.encoder_layers = _get_clones(encoder_layer, num_encoder)
        self.encoder_norm = nn.LayerNorm(d_model) if norm_before else None
        ## Transformer Decoder
        decoder_layer = PlainDETRTransformerDecoderLayer(d_model, decoder_num_head, decoder_mlp_ratio, decoder_dropout, decoder_act_type)
        self.decoder_layers = _get_clones(decoder_layer, num_decoder)
        self.decoder_norm = nn.LayerNorm(d_model)
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
            encoder_layer = PlainDETRTransformerEncoderLayer(d_model, encoder_num_head, encoder_mlp_ratio, encoder_dropout, encoder_act_type)
            self.encoder_layers = _get_clones(encoder_layer, num_encoder)
            self.encoder_norm = nn.LayerNorm(d_model) if norm_before else None
        ## Upsample layer
        self.upsample_layer = None
        if upsample:
            self.upsample_layer = nn.ConvTranspose2d(d_model, d_model, kernel_size=4, padding=1, stride=2)
        ## Transformer Decoder
        self.decoder_layers = None
        if num_decoder > 0:
            decoder_layer = PlainDETRTransformerDecoderLayer(d_model, decoder_num_head, decoder_mlp_ratio, decoder_dropout, decoder_act_type)
            self.decoder_layers = _get_clones(decoder_layer, num_decoder)
            self.decoder_norm = nn.LayerNorm(d_model)
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

    def pos2posembed(self, pos, temperature=10000):
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

    def get_posembed(self, x, temperature=10000):
        hs, ws = x.shape[-2:]
        scale = 2 * math.pi
        # generate xy coord mat
        y_embed, x_embed = torch.meshgrid(
            [torch.arange(1, hs+1, dtype=torch.float32),
             torch.arange(1, ws+1, dtype=torch.float32)])
        y_embed = y_embed / (hs + 1e-6) * scale
        x_embed = x_embed / (ws + 1e-6) * scale
    
        # [H, W] -> [1, H, W]
        y_embed = y_embed[None, :, :].to(x.device)
        x_embed = x_embed[None, :, :].to(x.device)

        # [1, H, W, 2]
        pos = torch.stack([x_embed, y_embed], dim=-1)
        # [1, H, W, C]
        pos_embed = self.pos2posembed(pos, temperature)
        pos_embed = pos_embed.permute(0, 3, 1, 2)

        # dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        # dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        # dim_t = temperature ** (2 * dim_t_)

        # pos_x = torch.div(x_embed[..., None], dim_t)
        # pos_y = torch.div(y_embed[..., None], dim_t)
        # pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        # pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        # # [B, H, W, C] -> [B, C, H, W]
        # pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos_embed        

    def inverse_sigmoid(self, x):
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=1e-5)/(1 - x).clamp(min=1e-5))

    def resize_mask(self, src, mask=None):
        bs, c, h, w = src.shape
        if mask is not None:
            # [B, H, W]
            mask = nn.functional.interpolate(mask[None].float(), size=[h, w]).bool()[0]
        else:
            mask = torch.zeros([bs, h, w], device=src.device, dtype=torch.bool)

        return mask

    def forward(self, src, src_mask=None):
        bs, c, h, w = src.shape
        mask = self.resize_mask(src, src_mask)
        mask = mask.flatten(1)

        # ------------------------ Transformer Encoder ------------------------
        ## Get pos_embed: [B, C, H, W]
        pos_embed = self.get_posembed(src)
        ## Reshape: [B, C, H, W] -> [N, B, C], N = HW
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        ## Encoder layer
        if self.encoder_layers:
            for encoder_layer in self.encoder_layers:
                src = encoder_layer(src, src_key_padding_mask=mask, pos_embed=pos_embed)

        ## Upsample feature
        if self.upsample_layer:
            # Reshape: [N, B, C] -> [B, C, H, W]
            src = src.permute(1, 2, 0).reshape(bs, c, h, w)
            src = self.upsample_layer(src)
            mask = self.resize_mask(src, src_mask)
            mask = mask.flatten(1)
            # Generate pos_embed for upsampled src
            pos_embed = self.get_posembed(src)
            ## Reshape: [B, C, H, W] -> [N, B, C], N = HW
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        # ------------------------ Transformer Decoder ------------------------
        ## Reshape tgt: [Nq, C] -> [Nq, B, C]
        tgt = self.query_embed.weight[:, None, :].repeat(1, bs, 1)
        rfp_embed = self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1)
        ref_point = rfp_embed.sigmoid()
        ref_points = [ref_point]
        
        ## Decoder layer
        outputs = []
        output_classes = []
        output_coords = []
        for layer_id, decoder_layer in enumerate(self.decoder_layers):
            # Adaptive pos embed
            query_pos = self.adapt_pos(self.pos2posembed(ref_point))
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
