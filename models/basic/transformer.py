import math
import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from ..basic.mlp import FFN
from ..basic.conv import LayerNorm2D
from ..basic.attn import GlobalCrossAttention


# ----------------- Basic Ops -----------------
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Copy from timm"""
    with torch.no_grad():
        """Copy from timm"""
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                        "The distribution of values may be incorrect.",
                        stacklevel=2)

        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)

        return tensor
    
def get_clones(module, N):
    if N <= 0:
        return None
    else:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0., max=1.)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))

def build_transformer(cfg, return_intermediate=False):
    if cfg['transformer'] == 'plain_detr_transformer':
        return PlainDETRTransformer(d_model             = cfg['hidden_dim'],
                                    num_heads           = cfg['de_num_heads'],
                                    ffn_dim             = cfg['de_ffn_dim'],
                                    dropout             = cfg['de_dropout'],
                                    act_type            = cfg['de_act'],
                                    pre_norm            = cfg['de_pre_norm'],
                                    rpe_hidden_dim      = cfg['rpe_hidden_dim'],
                                    feature_stride      = cfg['out_stride'],
                                    num_layers          = cfg['de_num_layers'],
                                    return_intermediate = return_intermediate,
                                    use_checkpoint      = cfg['use_checkpoint'],
                                    num_queries_one2one = cfg['num_queries_one2one'],
                                    num_queries_one2many    = cfg['num_queries_one2many'],
                                    proposal_feature_levels = cfg['proposal_feature_levels'],
                                    proposal_in_stride      = cfg['out_stride'],
                                    proposal_tgt_strides    = cfg['proposal_tgt_strides'],
                                    )
    

# ----------------- Transformer Encoder modules -----------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model   :int   = 256,
                 num_heads :int   = 8,
                 ffn_dim   :int   = 1024,
                 dropout   :float = 0.1,
                 act_type  :str   = "relu",
                 pre_norm  :bool = False,
                 ):
        super().__init__()
        # ----------- Basic parameters -----------
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.act_type = act_type
        self.pre_norm = pre_norm
        # ----------- Basic parameters -----------
        # Multi-head Self-Attn
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Feedforwaed Network
        self.ffn = FFN(d_model, ffn_dim, dropout, act_type)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre_norm(self, src, pos_embed):
        """
        Input:
            src:       [torch.Tensor] -> [B, N, C]
            pos_embed: [torch.Tensor] -> [B, N, C]
        Output:
            src:       [torch.Tensor] -> [B, N, C]
        """
        src = self.norm(src)
        q = k = self.with_pos_embed(src, pos_embed)

        # -------------- MHSA --------------
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout(src2)

        # -------------- FFN --------------
        src = self.ffn(src)
        
        return src

    def forward_post_norm(self, src, pos_embed):
        """
        Input:
            src:       [torch.Tensor] -> [B, N, C]
            pos_embed: [torch.Tensor] -> [B, N, C]
        Output:
            src:       [torch.Tensor] -> [B, N, C]
        """
        q = k = self.with_pos_embed(src, pos_embed)

        # -------------- MHSA --------------
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout(src2)
        src = self.norm(src)

        # -------------- FFN --------------
        src = self.ffn(src)
        
        return src

    def forward(self, src, pos_embed):
        if self.pre_norm:
            return self.forward_pre_norm(src, pos_embed)
        else:
            return self.forward_post_norm(src, pos_embed)

class TransformerEncoder(nn.Module):
    def __init__(self,
                 d_model        :int   = 256,
                 num_heads      :int   = 8,
                 num_layers     :int   = 1,
                 ffn_dim        :int   = 1024,
                 pe_temperature :float = 10000.,
                 dropout        :float = 0.1,
                 act_type       :str   = "relu",
                 pre_norm       :bool  = False,
                 ):
        super().__init__()
        # ----------- Basic parameters -----------
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.act_type = act_type
        self.pre_norm = pre_norm
        self.pe_temperature = pe_temperature
        self.pos_embed = None
        # ----------- Basic parameters -----------
        self.encoder_layers = get_clones(
            TransformerEncoderLayer(d_model, num_heads, ffn_dim, dropout, act_type, pre_norm), num_layers)

    def build_2d_sincos_position_embedding(self, device, w, h, embed_dim=256, temperature=10000.):
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        
        # ----------- Check cahed pos_embed -----------
        if self.pos_embed is not None and \
            self.pos_embed.shape[2:] == [h, w]:
            return self.pos_embed
        
        # ----------- Generate grid coords -----------
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid([grid_w, grid_h])  # shape: [H, W]

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None] # shape: [N, C]
        out_h = grid_h.flatten()[..., None] @ omega[None] # shape: [N, C]

        # shape: [1, N, C]
        pos_embed = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h),torch.cos(out_h)], dim=1)[None, :, :]
        pos_embed = pos_embed.to(device)
        self.pos_embed = pos_embed

        return pos_embed

    def forward(self, src):
        """
        Input:
            src:  [torch.Tensor] -> [B, C, H, W]
        Output:
            src:  [torch.Tensor] -> [B, C, H, W]
        """
        # -------- Transformer encoder --------
        channels, fmp_h, fmp_w = src.shape[1:]
        # [B, C, H, W] -> [B, N, C], N=HxW
        src_flatten = src.flatten(2).permute(0, 2, 1).contiguous()
        memory = src_flatten

        # PosEmbed: [1, N, C]
        pos_embed = self.build_2d_sincos_position_embedding(
            src.device, fmp_w, fmp_h, channels, self.pe_temperature)
        
        # Transformer Encoder layer
        for encoder in self.encoder_layers:
            memory = encoder(memory, pos_embed=pos_embed)

        # Output: [B, N, C] -> [B, C, N] -> [B, C, H, W]
        src = memory.permute(0, 2, 1).contiguous()
        src = src.view([-1, channels, fmp_h, fmp_w])

        return src


# ----------------- PlainDETR's Transformer Decoder -----------------
class GlobalDecoderLayer(nn.Module):
    def __init__(self,
                 d_model    :int   = 256,
                 num_heads  :int   = 8,
                 ffn_dim    :int = 1024,
                 dropout    :float = 0.1,
                 act_type   :str   = "relu",
                 pre_norm   :bool  = False,
                 rpe_hidden_dim :int = 512,
                 feature_stride :int = 16,
                 ) -> None:
        super().__init__()
        # ------------ Basic parameters ------------
        self.d_model = d_model
        self.num_heads = num_heads
        self.rpe_hidden_dim = rpe_hidden_dim
        self.ffn_dim = ffn_dim
        self.act_type = act_type
        self.pre_norm = pre_norm

        # ------------ Network parameters ------------
        ## Multi-head Self-Attn
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        ## Box-reparam Global Cross-Attn
        self.cross_attn = GlobalCrossAttention(d_model, num_heads, rpe_hidden_dim=rpe_hidden_dim, feature_stride=feature_stride)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        ## FFN
        self.ffn = FFN(d_model, ffn_dim, dropout, act_type, pre_norm)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre_norm(self,
                         tgt,
                         query_pos,
                         reference_points,
                         src,
                         src_pos_embed,
                         src_spatial_shapes,
                         src_padding_mask=None,
                         self_attn_mask=None,
                         ):
        # ----------- Multi-head self attention -----------
        tgt1 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt1, query_pos)
        tgt1 = self.self_attn(q.transpose(0, 1),        # [B, N, C] -> [N, B, C], batch_first = False
                              k.transpose(0, 1),        # [B, N, C] -> [N, B, C], batch_first = False
                              tgt1.transpose(0, 1),     # [B, N, C] -> [N, B, C], batch_first = False
                              attn_mask=self_attn_mask,
                              )[0].transpose(0, 1)      # [N, B, C] -> [B, N, C]
        tgt = tgt + self.dropout1(tgt1)

        # ----------- Global corss attention -----------
        tgt1 = self.norm2(tgt)
        tgt1 = self.cross_attn(self.with_pos_embed(tgt1, query_pos),
                               reference_points,
                               self.with_pos_embed(src, src_pos_embed),
                               src,
                               src_spatial_shapes,
                               src_padding_mask,
                               )
        tgt = tgt + self.dropout2(tgt1)

        # ----------- FeedForward Network -----------
        tgt = self.ffn(tgt)

        return tgt

    def forward_post_norm(self,
                          tgt,
                          query_pos,
                          reference_points,
                          src,
                          src_pos_embed,
                          src_spatial_shapes,
                          src_padding_mask=None,
                          self_attn_mask=None,
                          ):
        # ----------- Multi-head self attention -----------
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt1 = self.self_attn(q.transpose(0, 1),        # [B, N, C] -> [N, B, C], batch_first = False
                              k.transpose(0, 1),        # [B, N, C] -> [N, B, C], batch_first = False
                              tgt.transpose(0, 1),     # [B, N, C] -> [N, B, C], batch_first = False
                              attn_mask=self_attn_mask,
                              )[0].transpose(0, 1)      # [N, B, C] -> [B, N, C]
        tgt = tgt + self.dropout1(tgt1)
        tgt = self.norm1(tgt)

        # ----------- Global corss attention -----------
        tgt1 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               self.with_pos_embed(src, src_pos_embed),
                               src,
                               src_spatial_shapes,
                               src_padding_mask,
                               )
        tgt = tgt + self.dropout2(tgt1)
        tgt = self.norm2(tgt)

        # ----------- FeedForward Network -----------
        tgt = self.ffn(tgt)

        return tgt

    def forward(self,
                tgt,
                query_pos,
                reference_points,
                src,
                src_pos_embed,
                src_spatial_shapes,
                src_padding_mask=None,
                self_attn_mask=None,
                ):
        if self.pre_norm:
            return self.forward_pre_norm(tgt, query_pos, reference_points, src, src_pos_embed, src_spatial_shapes,
                                         src_padding_mask, self_attn_mask)
        else:
            return self.forward_post_norm(tgt, query_pos, reference_points, src, src_pos_embed, src_spatial_shapes,
                                          src_padding_mask, self_attn_mask)

class GlobalDecoder(nn.Module):
    def __init__(self,
                 # Decoder layer params
                 d_model    :int   = 256,
                 num_heads  :int   = 8,
                 ffn_dim    :int = 1024,
                 dropout    :float = 0.1,
                 act_type   :str   = "relu",
                 pre_norm   :bool  = False,
                 rpe_hidden_dim :int = 512,
                 feature_stride :int = 16,
                 num_layers     :int = 6,
                 # Decoder params
                 return_intermediate :bool = False,
                 use_checkpoint      :bool = False,
                 ):
        super().__init__()
        # ------------ Basic parameters ------------
        self.d_model = d_model
        self.num_heads = num_heads
        self.rpe_hidden_dim = rpe_hidden_dim
        self.ffn_dim = ffn_dim
        self.act_type = act_type
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.use_checkpoint = use_checkpoint

        # ------------ Network parameters ------------
        decoder_layer = GlobalDecoderLayer(
            d_model, num_heads, ffn_dim, dropout, act_type, pre_norm, rpe_hidden_dim, feature_stride,)
        self.layers = get_clones(decoder_layer, num_layers)
        self.bbox_embed = None
        self.class_embed = None

        if pre_norm:
            self.final_layer_norm = nn.LayerNorm(d_model)
        else:
            self.final_layer_norm = None

    def _reset_parameters(self):            
        # stolen from Swin Transformer
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def inverse_sigmoid(self, x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)

        return torch.log(x1 / x2)

    def box_xyxy_to_cxcywh(self, x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
        
        return torch.stack(b, dim=-1)

    def delta2bbox(self, proposals,
                   deltas,
                   max_shape=None,
                   wh_ratio_clip=16 / 1000,
                   clip_border=True,
                   add_ctr_clamp=False,
                   ctr_clamp=32):

        dxy = deltas[..., :2]
        dwh = deltas[..., 2:]

        # Compute width/height of each roi
        pxy = proposals[..., :2]
        pwh = proposals[..., 2:]

        dxy_wh = pwh * dxy
        wh_ratio_clip = torch.as_tensor(wh_ratio_clip)
        max_ratio = torch.abs(torch.log(wh_ratio_clip)).item()
        
        if add_ctr_clamp:
            dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
            dwh = torch.clamp(dwh, max=max_ratio)
        else:
            dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

        gxy = pxy + dxy_wh
        gwh = pwh * dwh.exp()
        x1y1 = gxy - (gwh * 0.5)
        x2y2 = gxy + (gwh * 0.5)
        bboxes = torch.cat([x1y1, x2y2], dim=-1)
        if clip_border and max_shape is not None:
            bboxes[..., 0::2].clamp_(min=0).clamp_(max=max_shape[1])
            bboxes[..., 1::2].clamp_(min=0).clamp_(max=max_shape[0])

        return bboxes

    def forward(self,
                tgt,
                reference_points,
                src,
                src_pos_embed,
                src_spatial_shapes,
                query_pos=None,
                src_padding_mask=None,
                self_attn_mask=None,
                max_shape=None,
                ):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None]
            if self.use_checkpoint:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    query_pos,
                    reference_points_input,
                    src,
                    src_pos_embed,
                    src_spatial_shapes,
                    src_padding_mask,
                    self_attn_mask,
                )
            else:
                output = layer(
                    output,
                    query_pos,
                    reference_points_input,
                    src,
                    src_pos_embed,
                    src_spatial_shapes,
                    src_padding_mask,
                    self_attn_mask,
                )

            if self.final_layer_norm is not None:
                output_after_norm = self.final_layer_norm(output)
            else:
                output_after_norm = output

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output_after_norm)
                new_reference_points = self.box_xyxy_to_cxcywh(
                    self.delta2bbox(reference_points, tmp, max_shape)) 
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output_after_norm)
                intermediate_reference_points.append(new_reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output_after_norm, reference_points


# ----------------- PlainDETR's Transformer -----------------
class PlainDETRTransformer(nn.Module):
    def __init__(self,
                 # Decoder layer params
                 d_model        :int   = 256,
                 num_heads      :int   = 8,
                 ffn_dim        :int   = 1024,
                 dropout        :float = 0.1,
                 act_type       :str   = "relu",
                 pre_norm       :bool  = False,
                 rpe_hidden_dim :int   = 512,
                 feature_stride :int   = 16,
                 num_layers     :int   = 6,
                 # Decoder params
                 return_intermediate     :bool = False,
                 use_checkpoint          :bool = False,
                 num_queries_one2one     :int  = 300,
                 num_queries_one2many    :int  = 1500,
                 proposal_feature_levels :int  = 3,
                 proposal_in_stride      :int  = 16,
                 proposal_tgt_strides    :int  = [8, 16, 32],
                 ):
        super().__init__()
        # ------------ Basic setting ------------
        ## Model
        self.d_model = d_model
        self.num_heads = num_heads
        self.rpe_hidden_dim = rpe_hidden_dim
        self.ffn_dim = ffn_dim
        self.act_type = act_type
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        ## Trick
        self.use_checkpoint = use_checkpoint
        self.num_queries_one2one = num_queries_one2one
        self.num_queries_one2many = num_queries_one2many
        self.proposal_feature_levels = proposal_feature_levels
        self.proposal_tgt_strides = proposal_tgt_strides
        self.proposal_in_stride = proposal_in_stride
        self.proposal_min_size = 50

        # --------------- Network setting ---------------
        ## Global Decoder
        self.decoder = GlobalDecoder(d_model, num_heads, ffn_dim, dropout, act_type, pre_norm,
                                     rpe_hidden_dim, feature_stride, num_layers, return_intermediate,
                                     use_checkpoint,)
        
        ## Two stage
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
        self.pos_trans_norm = nn.LayerNorm(d_model * 2)

        ## Expand layers
        if proposal_feature_levels > 1:
            assert len(proposal_tgt_strides) == proposal_feature_levels

            self.enc_output_proj = nn.ModuleList([])
            for stride in proposal_tgt_strides:
                if stride == proposal_in_stride:
                    self.enc_output_proj.append(nn.Identity())
                elif stride > proposal_in_stride:
                    scale = int(math.log2(stride / proposal_in_stride))
                    layers = []
                    for _ in range(scale - 1):
                        layers += [
                            nn.Conv2d(d_model, d_model, kernel_size=2, stride=2),
                            LayerNorm2D(d_model),
                            nn.GELU()
                        ]
                    layers.append(nn.Conv2d(d_model, d_model, kernel_size=2, stride=2))
                    self.enc_output_proj.append(nn.Sequential(*layers))
                else:
                    scale = int(math.log2(proposal_in_stride / stride))
                    layers = []
                    for _ in range(scale - 1):
                        layers += [
                            nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2),
                            LayerNorm2D(d_model),
                            nn.GELU()
                        ]
                    layers.append(nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2))
                    self.enc_output_proj.append(nn.Sequential(*layers))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if hasattr(self.decoder, '_reset_parameters'):
            print('decoder re-init')
            self.decoder._reset_parameters()

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = self.d_model // 2
        temperature = 10000
        scale = 2 * torch.pi

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device
        )
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)

        return pos

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)

        return valid_ratio

    def expand_encoder_output(self, memory, memory_padding_mask, spatial_shapes):
        assert spatial_shapes.size(0) == 1, f'Get encoder output of shape {spatial_shapes}, not sure how to expand'

        bs, _, c = memory.shape
        h, w = spatial_shapes[0]

        _out_memory = memory.view(bs, h, w, c).permute(0, 3, 1, 2)
        _out_memory_padding_mask = memory_padding_mask.view(bs, h, w)

        out_memory, out_memory_padding_mask, out_spatial_shapes = [], [], []
        for i in range(self.proposal_feature_levels):
            mem = self.enc_output_proj[i](_out_memory)
            mask = F.interpolate(
                _out_memory_padding_mask[None].float(), size=mem.shape[-2:]
            ).to(torch.bool)

            out_memory.append(mem)
            out_memory_padding_mask.append(mask.squeeze(0))
            out_spatial_shapes.append(mem.shape[-2:])

        out_memory = torch.cat([mem.flatten(2).transpose(1, 2) for mem in out_memory], dim=1)
        out_memory_padding_mask = torch.cat([mask.flatten(1) for mask in out_memory_padding_mask], dim=1)
        out_spatial_shapes = torch.as_tensor(out_spatial_shapes, dtype=torch.long, device=out_memory.device)
        
        return out_memory, out_memory_padding_mask, out_spatial_shapes

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        if self.proposal_feature_levels > 1:
            memory, memory_padding_mask, spatial_shapes = self.expand_encoder_output(
                memory, memory_padding_mask, spatial_shapes
            )
        N_, S_, C_ = memory.shape
        # base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            stride = self.proposal_tgt_strides[lvl]

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) * stride
            wh = torch.ones_like(grid) * self.proposal_min_size * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)

        H_, W_ = spatial_shapes[0]
        stride = self.proposal_tgt_strides[0]
        mask_flatten_ = memory_padding_mask[:, :H_*W_].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1, keepdim=True) * stride
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1, keepdim=True) * stride
        img_size = torch.cat([valid_W, valid_H, valid_W, valid_H], dim=-1)
        img_size = img_size.unsqueeze(1) # [BS, 1, 4]

        output_proposals_valid = (
            (output_proposals > 0.01 * img_size) & (output_proposals < 0.99 * img_size)
        ).all(-1, keepdim=True)
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1).repeat(1, 1, 1),
            max(H_, W_) * stride,
        )
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid,
            max(H_, W_) * stride,
        )

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        max_shape = (valid_H[:, None, :], valid_W[:, None, :])
        return output_memory, output_proposals, max_shape
    
    def get_reference_points(self, memory, mask_flatten, spatial_shapes):
        output_memory, output_proposals, max_shape = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )

        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_delta = self.decoder.bbox_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.decoder.box_xyxy_to_cxcywh(self.decoder.delta2bbox(
            output_proposals,
            enc_outputs_delta,
            max_shape
        ))

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact
        
        return (reference_points, max_shape, enc_outputs_class,
                enc_outputs_coord_unact, enc_outputs_delta, output_proposals)

    def forward(self, src, mask, pos_embed, query_embed=None, self_attn_mask=None):
        # Prepare input for encoder
        bs, c, h, w = src.shape
        src_flatten = src.flatten(2).transpose(1, 2)
        mask_flatten = mask.flatten(1)
        pos_embed_flatten = pos_embed.flatten(2).transpose(1, 2)
        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long, device=src_flatten.device)

        # Prepare input for decoder
        memory = src_flatten
        bs, _, c = memory.shape

        # Two stage trick
        if self.training:
            self.two_stage_num_proposals = self.num_queries_one2one + self.num_queries_one2many
        else:
            self.two_stage_num_proposals = self.num_queries_one2one
        (reference_points, max_shape, enc_outputs_class,
        enc_outputs_coord_unact, enc_outputs_delta, output_proposals) \
            = self.get_reference_points(memory, mask_flatten, spatial_shapes)
        init_reference_out = reference_points
        pos_trans_out = torch.zeros((bs, self.two_stage_num_proposals, 2*c), device=init_reference_out.device)
        pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(reference_points)))

        # Mixed selection trick
        tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)
        query_embed, _ = torch.split(pos_trans_out, c, dim=2)

        # Decoder
        hs, inter_references = self.decoder(tgt,
                                            reference_points,
                                            memory,
                                            pos_embed_flatten,
                                            spatial_shapes,
                                            query_embed,
                                            mask_flatten,
                                            self_attn_mask,
                                            max_shape
                                            )
        inter_references_out = inter_references

        return (hs,
                init_reference_out,
                inter_references_out,
                enc_outputs_class,
                enc_outputs_coord_unact,
                enc_outputs_delta,
                output_proposals,
                max_shape
                )
