# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter


model_urls = {
    # CLIP pretrained
    'resnet50_clip':  "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    'resnet101_clip': "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
}


# Frozen BatchNormazlizarion
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


# -------------------- ResNet series --------------------
class ResNet(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, pretrained: bool, res5_dilation: bool, norm_type: str):
        super().__init__()
        # norm layer
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'FrozeBN':
            norm_layer = FrozenBatchNorm2d
        # backbone
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, res5_dilation],
            pretrained=pretrained, norm_layer=norm_layer)
        feat_dims = [128, 256, 512] if name in ('resnet18', 'resnet34') else [512, 1024, 2048]
        # freeze 
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.feat_dims = feat_dims

    def forward(self, x):
        xs = self.body(x)
        fmp_list = []
        for name, fmp in xs.items():
            fmp_list.append(fmp)

        return fmp_list


# build backbone
def build_resnet(cfg, pretrained=False):
    # ResNet series
    backbone = ResNet(cfg['backbone'], pretrained, cfg['res5_dilation'], cfg['backbone_norm'])

    return backbone, backbone.feat_dims


if __name__ == '__main__':
    cfg = {
        'backbone':      'resnet50',
        'backbone_norm': 'FrozeBN',
        'pretrained':    True,
        'res5_dilation': False,
    }
    model, feat_dim = build_resnet(cfg, pretrained=True)
    print(feat_dim)

    x = torch.randn(2, 3, 320, 320)
    output = model(x)
    for y in output:
        print(y.size())
