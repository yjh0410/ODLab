from .resnet import build_resnet


def build_backbone(cfg, pretrained=False):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone']))
    
    if cfg['backbone'] in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        pretrained_weight = cfg['pretrained_weight'] if pretrained else None
        return build_resnet(cfg, pretrained_weight)
    
                           