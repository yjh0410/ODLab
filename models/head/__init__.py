from .retinanet_head import RetinaNetHead
from .yolof_head import YOLOFHead
from .fcos_head import FCOSHead
from .pfcos_head import PlainFCOSHead


# build head
def build_head(cfg, in_dim, out_dim, num_classes):
    print('==============================')
    print('Head: {}'.format(cfg['head']))
    
    if cfg['head'] == 'retinanet_head':
        model = RetinaNetHead(cfg          = cfg,
                              in_dim       = in_dim,
                              out_dim      = out_dim,
                              num_classes  = num_classes,
                              num_cls_head = cfg['num_cls_head'],
                              num_reg_head = cfg['num_reg_head'],
                              act_type     = cfg['head_act'],
                              norm_type    = cfg['head_norm']
                              )
    elif cfg['head'] == 'fcos_head':
        model = FCOSHead(cfg          = cfg,
                         in_dim       = in_dim,
                         out_dim      = out_dim,
                         num_classes  = num_classes,
                         num_cls_head = cfg['num_cls_head'],
                         num_reg_head = cfg['num_reg_head'],
                         act_type     = cfg['head_act'],
                         norm_type    = cfg['head_norm']
                         )
    elif cfg['head'] == 'pfcos_head':
        model = PlainFCOSHead(cfg          = cfg,
                              in_dim       = in_dim,
                              out_dim      = out_dim,
                              num_classes  = num_classes,
                              act_type     = cfg['head_act'],
                              norm_type    = cfg['head_norm']
                              )
    elif cfg['head'] == 'yolof_head':
        model = YOLOFHead(cfg          = cfg,
                          in_dim       = in_dim,
                          out_dim      = out_dim,
                          num_classes  = num_classes,
                          num_cls_head = cfg['num_cls_head'],
                          num_reg_head = cfg['num_reg_head'],
                          act_type     = cfg['head_act'],
                          norm_type    = cfg['head_norm']
                          )

    return model
