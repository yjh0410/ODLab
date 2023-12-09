from .dilated_encoder import DilatedEncoder
from .fpn import BasicFPN, PaFPN, DETRXPaFPN


# build neck
def build_neck(cfg, in_dim, out_dim):
    print('==============================')
    print('Neck: {}'.format(cfg['neck']))

    if cfg['neck'] == 'dilated_encoder':
        model = DilatedEncoder(in_dim       = in_dim,
                               out_dim      = out_dim,
                               expand_ratio = cfg['neck_expand_ratio'],
                               dilations    = cfg['neck_dilations'],
                               act_type     = cfg['neck_act'],
                               norm_type    = cfg['neck_norm']
                               )
    elif cfg['neck'] == 'basic_fpn':
        model = BasicFPN(in_dims = in_dim,
                         out_dim = out_dim,
                         p6_feat = cfg['fpn_p6_feat'],
                         p7_feat = cfg['fpn_p7_feat'],
                         from_c5 = cfg['fpn_p6_from_c5'], 
                         )
    elif cfg['neck'] == 'pafpn':
        model = PaFPN(in_dims = in_dim,
                      out_dim = out_dim,
                      p6_feat = cfg['fpn_p6_feat'],
                      p7_feat = cfg['fpn_p7_feat'],
                      )
    elif cfg['neck'] == 'detrx_pafpn':
        model = DETRXPaFPN(in_dims = in_dim,
                           out_dim = out_dim,
                           depth   = cfg['depth'],
                           p6_feat = cfg['fpn_p6_feat'],
                           p7_feat = cfg['fpn_p7_feat'],
                           from_p5 = False,
                           depthwise = cfg['fpn_depthwise']
                           )
        
    return model
