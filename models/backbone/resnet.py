# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import (ResNet18_Weights,
                                       ResNet34_Weights,
                                       ResNet50_Weights,
                                       ResNet101_Weights)

model_urls = {
    # IN1K-Cls pretrained weights
    'resnet18':  ResNet18_Weights,
    'resnet34':  ResNet34_Weights,
    'resnet50':  ResNet50_Weights,
    'resnet101': ResNet101_Weights,
}
spark_model_urls = {
    # SparK's IN1K-MAE pretrained weights
    'spark_resnet18': None,
    'spark_resnet34': None,
    'spark_resnet50': "https://github.com/yjh0410/RT-ODLab/releases/download/backbone_weight/resnet50_in1k_spark_pretrained_timm_style.pth",
    'spark_resnet101': None,
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
    """Standard ResNet backbone."""
    def __init__(self, name: str, res5_dilation: bool, norm_type: str, pretrained_weights: str = "imagenet1k_v1"):
        super().__init__()
        # Pretrained
        assert pretrained_weights in [None, "imagenet1k_v1", "imagenet1k_v2"]
        if pretrained_weights is not None:
            if name in ('resnet18', 'resnet34'):
                pretrained_weights = model_urls[name].IMAGENET1K_V1
            else:
                if pretrained_weights == "imagenet1k_v1":
                    pretrained_weights = model_urls[name].IMAGENET1K_V1
                else:
                    pretrained_weights = model_urls[name].IMAGENET1K_V2
        else:
            pretrained_weights = None
        print('ImageNet pretrained weight: ', pretrained_weights)
        # Norm layer
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'FrozeBN':
            norm_layer = FrozenBatchNorm2d
        # Backbone
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, res5_dilation],
            norm_layer=norm_layer, weights=pretrained_weights)
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.feat_dims = [128, 256, 512] if name in ('resnet18', 'resnet34') else [512, 1024, 2048]
        # Freeze
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

    def forward(self, x):
        xs = self.body(x)
        fmp_list = []
        for name, fmp in xs.items():
            fmp_list.append(fmp)

        return fmp_list

class SparkResNet(nn.Module):
    """ResNet backbone with SparK pretrained."""
    def __init__(self, name: str, res5_dilation: bool, norm_type: str, pretrained: bool = False):
        super().__init__()
        # Norm layer
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'FrozeBN':
            norm_layer = FrozenBatchNorm2d
        # Backbone
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, res5_dilation], norm_layer=norm_layer)
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.feat_dims = [128, 256, 512] if name in ('resnet18', 'resnet34') else [512, 1024, 2048]

        # Load pretrained
        if pretrained:
            self.load_pretrained(name)

        # Freeze
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

    def load_pretrained(self, name):
        url = model_urls["spark_" + name]
        if url is not None:
            print('Loading pretrained weight from : {}'.format(url))
            # checkpoint state dict
            checkpoint_state_dict = torch.hub.load_state_dict_from_url(
                url=url, map_location="cpu", check_hash=True)
            # model state dict
            model_state_dict = self.body.state_dict()
            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        checkpoint_state_dict.pop(k)
                else:
                    checkpoint_state_dict.pop(k)
                    print('Unused key: ', k)
            # load the weight
            self.body.load_state_dict(checkpoint_state_dict)
        else:
            print('No backbone pretrained for {}.'.format(name))

    def forward(self, x):
        xs = self.body(x)
        fmp_list = []
        for name, fmp in xs.items():
            fmp_list.append(fmp)

        return fmp_list


# build backbone
def build_resnet(cfg, pretrained_weight=None):
    # ResNet series
    if pretrained_weight in spark_model_urls.keys():
        backbone = SparkResNet(cfg['backbone'], cfg['res5_dilation'], cfg['backbone_norm'])
    else:
        backbone = ResNet(cfg['backbone'], cfg['res5_dilation'], cfg['backbone_norm'], pretrained_weight)

    return backbone, backbone.feat_dims


if __name__ == '__main__':
    cfg = {
        'backbone':      'resnet50',
        'backbone_norm': 'FrozeBN',
        'pretrained_weight': 'imagenet1k_v1',
        'res5_dilation': False,
    }
    model, feat_dim = build_resnet(cfg, cfg['pretrained_weight'])
    print(feat_dim)

    x = torch.randn(2, 3, 320, 320)
    output = model(x)
    for y in output:
        print(y.size())
