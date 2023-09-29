from .dilated_encoder import DilatedEncoder
from .fpn import BasicFPN


# build neck
def build_neck(cfg, in_dim, out_dim):
    if 'neck' in cfg.keys():
        model_type = 'neck'
    elif 'encoder' in cfg.keys():
        model_type = 'encoder'
        
    print('==============================')
    print('Neck: {}'.format(cfg[model_type]))
    
    if cfg[model_type] == 'dilated_encoder':
        model = DilatedEncoder(in_dim       = in_dim,
                               out_dim      = out_dim,
                               expand_ratio = cfg['encoder_expand_ratio'],
                               dilations    = cfg['encoder_dilations'],
                               act_type     = cfg[model_type + '_act'],
                               norm_type    = cfg[model_type + '_norm']
                               )
    elif cfg[model_type] == 'basic_fpn':
        model = BasicFPN(in_dims = in_dim,
                         out_dim = out_dim,
                         p6_feat = cfg['fpn_p6_feat'],
                         p7_feat = cfg['fpn_p7_feat'],
                         from_c5 = cfg['fpn_p6_from_c5'], 
                         )
        
    return model
