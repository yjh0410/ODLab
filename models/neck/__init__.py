from .dilated_encoder import DilatedEncoder
from .fpn import BasicFPN, FcosRTPaFPN, HybridEncoder
from .spp import SPPF


# build neck
def build_neck(cfg, in_dim, out_dim):
    print('==============================')
    print('Neck: {}'.format(cfg['neck']))

    # ----------------------- Neck module -----------------------
    if cfg['neck'] == 'dilated_encoder':
        model = DilatedEncoder(in_dim       = in_dim,
                               out_dim      = out_dim,
                               expand_ratio = cfg['neck_expand_ratio'],
                               dilations    = cfg['neck_dilations'],
                               act_type     = cfg['neck_act'],
                               norm_type    = cfg['neck_norm']
                               )
    elif cfg['neck'] == 'spp_block':
        model = SPPF(in_dim       = in_dim,
                     out_dim      = out_dim,
                     expand_ratio = cfg['neck_expand_ratio'],
                     pooling_size = cfg["spp_pooling_size"],
                     act_type     = cfg['neck_act'],
                     norm_type    = cfg['neck_norm']
                     )
        
    # ----------------------- FPN Neck -----------------------
    elif cfg['neck'] == 'basic_fpn':
        model = BasicFPN(in_dims = in_dim,
                         out_dim = out_dim,
                         p6_feat = cfg['fpn_p6_feat'],
                         p7_feat = cfg['fpn_p7_feat'],
                         from_c5 = cfg['fpn_p6_from_c5'], 
                         )
    elif cfg['neck'] == 'fcos_rt_pafpn':
        if cfg['use_spp']:
            spp_block = SPPF(out_dim, out_dim,
                             expand_ratio=0.5, pooling_size=cfg["spp_pooling_size"],
                             act_type=cfg["spp_act"], norm_type=cfg["spp_norm"])
        else:
            spp_block = None
        model = FcosRTPaFPN(in_dims = in_dim,
                            out_dim = out_dim,
                            depth   = cfg['depth'],
                            spp_block = spp_block,
                            act_type  = cfg['fpn_act'],
                            norm_type = cfg['fpn_norm'],
                            depthwise = cfg['fpn_depthwise']
                            )
    elif cfg['fpn'] == 'hybrid_encoder':
        model = HybridEncoder(in_dims     = in_dim,
                              out_dim     = out_dim,
                              num_blocks  = cfg['fpn_num_blocks'],
                              act_type    = cfg['fpn_act'],
                              norm_type   = cfg['fpn_norm'],
                              depthwise   = cfg['fpn_depthwise'],
                              num_heads   = cfg['en_num_heads'],
                              num_layers  = cfg['en_num_layers'],
                              ffn_dim     = cfg['en_ffn_dim'],
                              dropout     = cfg['en_dropout'],
                              pe_temperature = cfg['pe_temperature'],
                              en_act_type    = cfg['en_act'],
                              )
    else:
        raise NotImplementedError("Unknown PaFPN: <{}>".format(cfg['fpn']))
        
    return model
