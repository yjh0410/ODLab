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

from ..basic.mlp import FFN, MLP


# ---------------------------- Basic functions ----------------------------
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# ---------------------------- Transformer Decoder layer ----------------------------
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


# ---------------------------- Transformer Decoder ----------------------------
class DETRTransformerDecoder(nn.Module):
    def __init__(self,
                 d_model             :int   = 512,
                 # Decoder
                 num_decoder         :int   = 6,
                 decoder_num_head    :int   = 8,
                 decoder_mlp_ratio   :float = 4.0,
                 decoder_dropout     :float = 0.1,
                 decoder_act_type    :str   = "relu",
                 # Other
                 num_classes          :int   = 80,
                 num_queries          :int   = 300,
                 norm_before          :bool  = False,
                 return_intermediate  :bool  = False
                 ):
        super().__init__()
        # --------------- Basic parameters ---------------
        self.d_model = d_model
        self.scale = 2 * math.pi
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.return_intermediate = return_intermediate
        # --------------- Network parameters ---------------
        ## Transformer Decoder
        self.decoder_layers = None
        if num_decoder > 0:
            decoder_layer = DETRTransformerDecoderLayer(d_model, decoder_num_head, decoder_mlp_ratio, decoder_dropout, decoder_act_type)
            self.decoder_layers = _get_clones(decoder_layer, num_decoder)

        ## Adaptive pos_embed
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)

        ## Object Queries
        self.query_embed = nn.Embedding(self.num_queries, d_model)
        
        ## Output head
        self.class_embed = nn.Linear(self.d_model, num_classes)
        self.bbox_embed  = MLP(self.d_model, self.d_model, 4, 3)

        self.init_weight()

    # -------------- Basic functions --------------
    def init_weight(self):
        # init all layer weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # init class embed bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        # init bbox embed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    @torch.jit.unused
    def set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    # -------------- Model forward --------------
    def forward(self, src, src_mask=None, pos_embed=None, is_train=False):
        """
            src:       (torch.Tensor) [N, B, C]
            src_mask:  (torch.Tensor) [B, N]
            pos_embed: (torch.Tensor) [N, B, C]
            is_train:  (bool)
        """
        # ------------------------ Transformer Decoder ------------------------
        bs = src.shape[1]
        ## Prepare queries
        query_pos = self.query_embed.weight
        query_pos = query_pos[:, None, :].repeat(1, bs, 1)
        tgt = torch.zeros_like(query_pos)
                
        ## Decoder layer
        output = tgt
        outputs = []
        output_classes = []
        output_coords = []
        for decoder_layer in self.decoder_layers:
            # Decoder
            output = decoder_layer(output,
                                   src,
                                   memory_key_padding_mask = src_mask,
                                   pos                     = pos_embed,
                                   query_pos               = query_pos
                                   )
            ## class pred
            output_class = self.class_embed(output)
            ## bbox pred
            output_coord = self.bbox_embed(output).sigmoid()

            output_classes.append(output_class)
            output_coords.append(output_coord)

        # --------------------- Re-organize outputs ---------------------
        outputs = {
            "pred_logits": output_classes[-1],
            "pred_boxes":  output_coords[-1]
        }
        if self.return_intermediate:
            outputs['aux_outputs'] = self.set_aux_loss(output_classes, output_coords)

        return outputs

class DETRXTransformerDecoder(nn.Module):
    def __init__(self,
                 d_model             :int   = 512,
                 # Decoder
                 num_decoder         :int   = 6,
                 decoder_num_head    :int   = 8,
                 decoder_mlp_ratio   :float = 4.0,
                 decoder_dropout     :float = 0.1,
                 decoder_act_type    :str   = "relu",
                 # Other
                 num_classes          :int   = 80,
                 num_queries_one2one  :int   = 300,
                 num_queries_one2many :int   = 1500,
                 norm_before          :bool  = False,
                 look_forward_twice   :bool  = False,
                 return_intermediate  :bool  = False
                 ):
        super().__init__()
        # --------------- Basic parameters ---------------
        self.d_model = d_model
        self.scale = 2 * math.pi
        self.num_queries_one2one = num_queries_one2one
        self.num_queries_one2many = num_queries_one2many
        self.num_queries = num_queries_one2one + num_queries_one2many
        self.num_classes = num_classes
        self.look_forward_twice = look_forward_twice
        self.return_intermediate = return_intermediate
        # --------------- Network parameters ---------------
        ## Transformer Decoder
        self.decoder_layers = None
        if num_decoder > 0:
            decoder_layer = DETRTransformerDecoderLayer(d_model, decoder_num_head, decoder_mlp_ratio, decoder_dropout, decoder_act_type)
            self.decoder_layers = _get_clones(decoder_layer, num_decoder)

        ## Adaptive pos_embed
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)

        ## Object Queries
        self.query_embed = nn.Embedding(self.num_queries, d_model)
        self.refpoint_embed = nn.Embedding(self.num_queries, 4)
        
        ## Output head
        class_embed = nn.Linear(self.d_model, num_classes)
        bbox_embed  = MLP(self.d_model, self.d_model, 4, 3)
        self.class_embed = nn.ModuleList([copy.deepcopy(class_embed) for _ in range(num_decoder)])
        self.bbox_embed  = nn.ModuleList([copy.deepcopy(bbox_embed)  for _ in range(num_decoder)])

        self.init_weight()

    # -------------- Basic functions --------------
    def init_weight(self):
        # init all layer weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # init class embed bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        for class_embed in self.class_embed:
            class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        # init bbox embed
        for bbox_embed in self.bbox_embed:
            nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        # init refpoint xy
        self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
        self.refpoint_embed.weight.data[:, :2] = self.inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])

    def pos2posembed(self, pos, temperature=10000):
        num_pos_feats = self.d_model // 2

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)

        # Position embedding for XY
        x_embed = pos[..., 0] * self.scale
        y_embed = pos[..., 1] * self.scale
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        
        # Position embedding for WH
        if pos.size(-1) == 4:
            w_embed = pos[..., 2] * self.scale
            h_embed = pos[..., 3] * self.scale
            pos_w = w_embed[..., None] / dim_t
            pos_h = h_embed[..., None] / dim_t
            pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1).flatten(-2)
            pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1).flatten(-2)
            posemb = torch.cat((posemb, pos_w, pos_h), dim=-1)
        
        return posemb

    def get_posembed(self, mask, temperature=10000):
        not_mask = ~mask
        # [B, H, W]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + 1e-6)
        x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + 1e-6)
    
        # [H, W] -> [B, H, W, 2]
        pos = torch.stack([x_embed, y_embed], dim=-1)

        # [B, H, W, C]
        pos_embed = self.pos2posembed(pos, temperature)
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        
        return pos_embed        

    def inverse_sigmoid(self, x):
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=1e-5)/(1 - x).clamp(min=1e-5))

    @torch.jit.unused
    def set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    # -------------- Model forward --------------
    def forward(self, src, src_mask=None, pos_embed=None, is_train=False):
        """
            src:       (torch.Tensor) [N, B, C]
            src_mask:  (torch.Tensor) [B, N]
            pos_embed: (torch.Tensor) [N, B, C]
            is_train:  (bool)
        """
        # ------------------------ Transformer Decoder ------------------------
        bs = src.shape[1]
        ## Prepare queries
        tgt = self.query_embed.weight
        refpoint_embed = self.refpoint_embed.weight
        tgt = tgt[:, None, :].repeat(1, bs, 1)
        refpoint_embed = refpoint_embed[:, None, :].repeat(1, bs, 1)
        
        ## Prepare attn mask
        use_one2many = self.num_queries_one2many > 0 and is_train
        if use_one2many:
            self_attn_mask = torch.zeros([self.num_queries, self.num_queries]).bool().to(src.device)
            self_attn_mask[self.num_queries_one2one:, :self.num_queries_one2one] = True
            self_attn_mask[:self.num_queries_one2one, self.num_queries_one2one:] = True
        else:
            self_attn_mask = None
            tgt = tgt[:self.num_queries_one2one]
            refpoint_embed = refpoint_embed[:self.num_queries_one2one]

        ## Initial reference points
        ref_point = refpoint_embed.sigmoid()
        ref_points = [ref_point]
        
        ## Decoder layer
        output = tgt
        outputs = []
        for layer_id, decoder_layer in enumerate(self.decoder_layers):
            # Conditional query
            query_sine_embed = self.pos2posembed(ref_point)
            query_pos = self.ref_point_head(query_sine_embed)

            # Decoder
            output = decoder_layer(tgt                     = output,
                                   memory                  = src,
                                   tgt_mask                = self_attn_mask,
                                   memory_key_padding_mask = src_mask,
                                   pos                     = pos_embed,
                                   query_pos               = query_pos
                                   )
            
            # Iter update
            tmp = self.bbox_embed[layer_id](output)
            new_ref_point = tmp + self.inverse_sigmoid(ref_point)
            new_ref_point = new_ref_point.sigmoid()
            ref_point = new_ref_point.detach()

            outputs.append(output)
            ref_points.append(new_ref_point if self.look_forward_twice else ref_point)

        # ------------------------ Detection Head ------------------------
        ## collect one2one outputs
        output_classes_one2one = []
        output_coords_one2one = []
        ## collect one2many outputs
        output_classes_one2many = []
        output_coords_one2many = []
        for lid, (ref_point, output) in enumerate(zip(ref_points[:-1], outputs)):
            ## class pred
            output_class = self.class_embed[lid](output)
            ## bbox pred
            tmp = self.bbox_embed[lid](output)
            tmp += self.inverse_sigmoid(ref_point)
            output_coord = tmp.sigmoid()

            output_classes_one2one.append(output_class[:self.num_queries_one2one])
            output_coords_one2one.append(output_coord[:self.num_queries_one2one])
            if use_one2many:
                output_classes_one2many.append(output_class[self.num_queries_one2one:])
                output_coords_one2many.append(output_coord[self.num_queries_one2one:])

        # --------------------- Re-organize outputs ---------------------
        ## One2one outputs
        # [L, Nq, B, Nc] -> [L, B, Nq, Nc]
        output_classes_one2one = torch.stack(output_classes_one2one).permute(0, 2, 1, 3)
        output_coords_one2one  = torch.stack(output_coords_one2one).permute(0, 2, 1, 3)
        outputs = {
            "pred_logits": output_classes_one2one[-1],
            "pred_boxes":  output_coords_one2one[-1],
        }
        if self.return_intermediate:
            outputs['aux_outputs'] = self.set_aux_loss(output_classes_one2one, output_coords_one2one)
        ## One2many outputs
        if use_one2many:
            output_classes_one2many = torch.stack(output_classes_one2many).permute(0, 2, 1, 3)
            output_coords_one2many  = torch.stack(output_coords_one2many).permute(0, 2, 1, 3)
            outputs["pred_logits_one2many"] = output_classes_one2many[-1]
            outputs["pred_boxes_one2many"]  = output_coords_one2many[-1]
            if self.return_intermediate:
                outputs['aux_outputs_one2many'] = self.set_aux_loss(output_classes_one2many, output_coords_one2many)

        return outputs
