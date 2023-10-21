from .resnet import build_resnet
from .swin_transformer import built_swin_transformer


def build_backbone(cfg, pretrained=False):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone']))
    # ResNet
    if cfg['backbone'] in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        pretrained_weight = cfg['pretrained_weight'] if pretrained else None
        return build_resnet(cfg, pretrained_weight)
    # SwinTransformer
    elif cfg['backbone'] in ['swin_t', 'swin_s', 'swin_b', 'swin_l']:
        return built_swin_transformer(cfg, pretrained)
    
                           