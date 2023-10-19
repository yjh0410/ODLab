from .transformer import DETRTransformer, PlainDETRTransformer


def build_transformer(cfg, num_classes, return_intermediate=False):
    if cfg['transformer'] == "detr_transformer":
        return DETRTransformer(d_model             = cfg['d_model'],
                               num_encoder         = cfg['num_encoder'],
                               encoder_num_head    = cfg['encoder_num_head'],
                               encoder_mlp_ratio   = cfg['encoder_mlp_ratio'],
                               encoder_dropout     = cfg['encoder_dropout'],
                               encoder_act_type    = cfg['encoder_act'],
                               num_decoder         = cfg['num_decoder'],
                               decoder_num_head    = cfg['decoder_num_head'],
                               decoder_mlp_ratio   = cfg['decoder_mlp_ratio'],
                               decoder_dropout     = cfg['decoder_dropout'],
                               decoder_act_type    = cfg['decoder_act'],
                               num_classes         = num_classes,
                               return_intermediate = return_intermediate
                               )
    elif cfg['transformer'] == "plain_detr_transformer":
        return PlainDETRTransformer(d_model             = cfg['d_model'],
                                    num_encoder         = cfg['num_encoder'],
                                    encoder_num_head    = cfg['encoder_num_head'],
                                    encoder_mlp_ratio   = cfg['encoder_mlp_ratio'],
                                    encoder_dropout     = cfg['encoder_dropout'],
                                    encoder_act_type    = cfg['encoder_act'],
                                    upsample            = cfg['upsample_c5'],
                                    num_decoder         = cfg['num_decoder'],
                                    decoder_num_head    = cfg['decoder_num_head'],
                                    decoder_mlp_ratio   = cfg['decoder_mlp_ratio'],
                                    decoder_dropout     = cfg['decoder_dropout'],
                                    decoder_act_type    = cfg['decoder_act'],
                                    num_classes         = num_classes,
                                    num_queries         = cfg['num_queries'],
                                    return_intermediate = return_intermediate
                                    )
    else:
        return
