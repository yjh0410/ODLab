from .retinanet_head import RetinaNetHead
from .yolof_head import YOLOFHead
from .fcos_head import FCOSHead


# build head
def build_head(cfg, in_dim, out_dim, num_classes):
    if 'head' in cfg.keys():
        model_type = 'head'
    elif 'decoder' in cfg.keys():
        model_type = 'decoder'
        
    print('==============================')
    print('Head: {}'.format(cfg[model_type]))
    
    if cfg[model_type] == 'retinanet_head':
        model = RetinaNetHead(cfg          = cfg,
                              in_dim       = in_dim,
                              out_dim      = out_dim,
                              num_classes  = num_classes,
                              num_cls_head = cfg['num_cls_head'],
                              num_reg_head = cfg['num_reg_head'],
                              act_type     = cfg[model_type + '_act'],
                              norm_type    = cfg[model_type + '_norm']
                              )
    elif cfg[model_type] == 'fcos_head':
        model = FCOSHead(cfg          = cfg,
                         in_dim       = in_dim,
                         out_dim      = out_dim,
                         num_classes  = num_classes,
                         num_cls_head = cfg['num_cls_head'],
                         num_reg_head = cfg['num_reg_head'],
                         act_type     = cfg[model_type + '_act'],
                         norm_type    = cfg[model_type + '_norm']
                         )

    elif cfg[model_type] == 'yolof_head':
        model = YOLOFHead(cfg          = cfg,
                          in_dim       = in_dim,
                          out_dim      = out_dim,
                          num_classes  = num_classes,
                          num_cls_head = cfg['num_cls_head'],
                          num_reg_head = cfg['num_reg_head'],
                          act_type     = cfg[model_type + '_act'],
                          norm_type    = cfg[model_type + '_norm']
                          )

    return model
