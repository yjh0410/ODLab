import math
import torch
import torch.nn as nn

from ...backbone import build_backbone
from ...neck import build_neck
from ...transformer import build_transformer_encoder, build_transformer_decoder


# Enhanced DETR
class DETRX(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes :int   = 80, 
                 topk        :int   = 100,
                 trainable   :bool  = False,
                 aux_loss    :bool  = False):
        super(DETRX, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.cfg         = cfg
        self.device      = device
        self.trainable   = trainable
        self.num_topk    = topk
        self.num_classes = num_classes
        self.aux_loss    = aux_loss
        self.scale       = 2 * math.pi
        self.d_model     = cfg['d_model']
        self.max_stride  = cfg['max_stride']
        self.out_stride  = cfg['out_stride']

        # ---------------------- Network Parameters ----------------------
        ## Backbone
        self.backbone, self.feat_dims = build_backbone(cfg, trainable&cfg['pretrained'])
        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(feat_dim, cfg['d_model'], kernel_size=1),
                          nn.GroupNorm(32, cfg['d_model'])
            ) for feat_dim in self.feat_dims
        )
        self.feat_dims = [cfg['d_model']] * len(self.feat_dims)
        ## Neck
        self.transformer_encoder = build_transformer_encoder(cfg)
        self.fpn = build_neck(cfg, self.feat_dims, cfg['d_model'])

        ## Decoder
        self.transformer_decoder = build_transformer_decoder(cfg, num_classes, return_intermediate=aux_loss)

    # ------------------- Basic functions -------------------
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

    def decode_bboxes(self, reg_preds, bbox_encode=False):
        if not bbox_encode:
            box_preds_x1y1 = reg_preds[..., :2] - 0.5 * reg_preds[..., 2:]
            box_preds_x2y2 = reg_preds[..., :2] + 0.5 * reg_preds[..., 2:]
            box_preds = torch.cat([box_preds_x1y1, box_preds_x2y2], dim=-1)

        return box_preds
    
    def post_process(self, cls_pred, box_pred):
        ## Top-k select
        cls_pred = cls_pred[0].flatten().sigmoid_()
        box_pred = box_pred[0]
        predicted_prob, topk_idxs = cls_pred.sort(descending=True)
        topk_idxs = topk_idxs[:self.num_topk]
        topk_box_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
        ## Top-k results
        topk_scores = predicted_prob[:self.num_topk]
        topk_labels = topk_idxs % self.num_classes
        topk_bboxes = box_pred[topk_box_idxs]

        return topk_bboxes, topk_scores, topk_labels

    @torch.no_grad()
    def inference_single_image(self, x):
        # ---------------- Backbone ----------------
        backbone_feats = self.backbone(x)
        pyramid_feats = []
        for feat, layer in zip(backbone_feats, self.input_proj):
            pyramid_feats.append(layer(feat))

        # ---------------- Encoder ----------------
        feat = pyramid_feats[-1].flatten(2).permute(2, 0, 1)
        bs, c, h, w = pyramid_feats[-1].shape
        mask = torch.zeros([bs, h, w], device=x.device, dtype=torch.bool)
        pos_embed = self.get_posembed(mask)
        feat = self.transformer_encoder(feat, mask.flatten(1), pos_embed.flatten(2).permute(2, 0, 1))
        pyramid_feats[-1] = feat.permute(1, 2, 0).reshape(bs, c, h, w)

        # ---------------- FPN ----------------
        pyramid_feats = self.fpn(pyramid_feats)
        memory_feats = []
        memory_masks = []
        memory_pos_embeds = []
        for feat in pyramid_feats:
            bs, c, h, w = feat.shape
            mask = torch.zeros([bs, h, w], device=x.device, dtype=torch.bool)
            pos_embed = self.get_posembed(mask)

            memory_feats.append(feat.flatten(2).permute(2, 0, 1))           # [N, B, C]
            memory_masks.append(mask.flatten(1))                            # [B, N]
            memory_pos_embeds.append(pos_embed.flatten(2).permute(2, 0, 1)) # [N, B, C]

        # ---------------- Decoder ----------------
        memory = torch.cat(memory_feats, dim=0)
        memory_mask = torch.cat(memory_masks, dim=1)
        memory_pos_embed = torch.cat(memory_pos_embeds, dim=0)
        outputs = self.transformer_decoder(memory, src_mask=memory_mask, pos_embed=memory_pos_embed)

        # ---------------- PostProcess ----------------
        cls_preds = outputs["pred_logits"]
        box_preds = self.decode_bboxes(outputs["pred_boxes"])
        bboxes, scores, labels = self.post_process(cls_preds, box_preds)

        return bboxes, scores, labels

    def forward(self, x, mask=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # ---------------- Backbone ----------------
            backbone_feats = self.backbone(x)
            pyramid_feats = []
            for feat, layer in zip(backbone_feats, self.input_proj):
                pyramid_feats.append(layer(feat))

            # ---------------- Encoder ----------------
            feat = pyramid_feats[-1].flatten(2).permute(2, 0, 1)
            bs, c, h, w = pyramid_feats[-1].shape
            mask = torch.zeros([bs, h, w], device=x.device, dtype=torch.bool)
            pos_embed = self.get_posembed(mask)
            feat = self.transformer_encoder(feat, mask.flatten(1), pos_embed.flatten(2).permute(2, 0, 1))
            pyramid_feats[-1] = feat.permute(1, 2, 0).reshape(bs, c, h, w)

            # ---------------- FPN ----------------
            pyramid_feats = self.fpn(pyramid_feats)
            memory_feats = []
            memory_masks = []
            memory_pos_embeds = []
            for feat in pyramid_feats:
                bs, c, h, w = feat.shape
                mask = torch.zeros([bs, h, w], device=x.device, dtype=torch.bool)
                pos_embed = self.get_posembed(mask)

                memory_feats.append(feat.flatten(2).permute(2, 0, 1))           # [N, B, C]
                memory_masks.append(mask.flatten(1))                            # [B, N]
                memory_pos_embeds.append(pos_embed.flatten(2).permute(2, 0, 1)) # [N, B, C]

            # ---------------- Decoder ----------------
            memory = torch.cat(memory_feats, dim=0)
            memory_mask = torch.cat(memory_masks, dim=1)
            memory_pos_embed = torch.cat(memory_pos_embeds, dim=0)
            outputs = self.transformer_decoder(memory, src_mask=memory_mask, pos_embed=memory_pos_embed, is_train=True)
            
            return outputs
