from .resnet import build_resnet


def build_backbone(cfg, pretrained=False):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone']))
    
    if cfg['backbone'] in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        return build_resnet(cfg, pretrained)
    
                           