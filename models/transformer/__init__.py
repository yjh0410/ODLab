from .transformer_encoder import DETRTransformerEncoder
from .transformer_decoder import DETRTransformerDecoder, DETRXTransformerDecoder


# ---------------------- Build Transformer Encoder ----------------------
def build_transformer_encoder(cfg):
    if cfg['transformer_encoder'] == "detr_encoder":
        return DETRTransformerEncoder(d_model            = cfg['d_model'],
                                      num_encoder        = cfg['num_encoder'],
                                      encoder_num_head   = cfg['encoder_num_head'],
                                      encoder_mlp_ratio  = cfg['encoder_mlp_ratio'],
                                      encoder_dropout    = cfg['encoder_dropout'],
                                      encoder_act_type   = cfg['encoder_act'],
                                      )


# ---------------------- Build Transformer Decoder ----------------------
def build_transformer_decoder(cfg, num_classes, return_intermediate=False):
    if cfg['transformer_decoder'] == "detr_decoder":
        return DETRTransformerDecoder(d_model            = cfg['d_model'],
                                      num_decoder         = cfg['num_decoder'],
                                      decoder_num_head    = cfg['decoder_num_head'],
                                      decoder_mlp_ratio   = cfg['decoder_mlp_ratio'],
                                      decoder_dropout     = cfg['decoder_dropout'],
                                      decoder_act_type    = cfg['decoder_act'],
                                      num_queries         = cfg['num_queries_one2one'],
                                      num_classes         = num_classes,
                                      return_intermediate = return_intermediate
                                      )
    elif cfg['transformer_decoder'] == "detrx_decoder":
        return DETRXTransformerDecoder(d_model              = cfg['d_model'],
                                       num_decoder          = cfg['num_decoder'],
                                       decoder_num_head     = cfg['decoder_num_head'],
                                       decoder_mlp_ratio    = cfg['decoder_mlp_ratio'],
                                       decoder_dropout      = cfg['decoder_dropout'],
                                       decoder_act_type     = cfg['decoder_act'],
                                       num_classes          = num_classes,
                                       num_queries_one2one  = cfg['num_queries_one2one'],
                                       num_queries_one2many = cfg['num_queries_one2many'],
                                       look_forward_twice   = cfg['look_forward_twice'],
                                       return_intermediate  = return_intermediate
                                       )
