import torch
import torch.nn as nn


def get_conv2d(c1, c2, k, p, s, d, g):
    conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g)

    return conv

def get_activation(act_type=None):
    if act_type is None:
        return nn.Identity()
    elif act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)

def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    elif norm_type is None:
        return nn.Identity()

class ConvModule(nn.Module):
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 p=0,
                 s=1,
                 d=1,
                 act_type='relu',
                 norm_type='BN', 
                 depthwise=False):
        super(ConvModule, self).__init__()
        convs = []
        if depthwise:
            convs.append(get_conv2d(c1, c1, k=k, p=p, s=s, d=d, g=c1))
            # depthwise conv
            if norm_type:
                convs.append(get_norm(norm_type, c1))
            if act_type:
                convs.append(get_activation(act_type))
            # pointwise conv
            convs.append(get_conv2d(c1, c2, k=1, p=0, s=1, d=d, g=1))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))

        else:
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=1))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)

class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim     :int,
                 out_dim    :int,
                 shortcut   :bool = False,
                 act_type   :str  = "relu",
                 norm_type  :str  = "BN",
                 depthwise  :bool = False):
        super(Bottleneck, self).__init__()
        self.cv1 = ConvModule(in_dim, out_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.cv2 = ConvModule(out_dim, out_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h

class CSPResBlock(nn.Module):
    def __init__(self,
                 in_dim    :int,
                 out_dim   :int,
                 depth     :int  = 1,
                 shortcut  :bool = False,
                 act_type  :str  = "relu",
                 norm_type :str  = "BN",
                 depthwise :bool = False
                 ):
        super(CSPResBlock, self).__init__()
        self.cv1 = ConvModule(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = ConvModule(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv3 = ConvModule(out_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.Sequential(*[Bottleneck(in_dim    = out_dim,
                                            out_dim   = out_dim,
                                            shortcut  = shortcut,
                                            act_type  = act_type,
                                            norm_type = norm_type,
                                            depthwise = depthwise
                                            )
                                for _ in range(depth)
                                ])

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.m(self.cv2(x))

        return self.cv3(x1 + x2)
    
class ELANBlock(nn.Module):
    def __init__(self,
                 in_dim    :int,
                 out_dim   :int,
                 depth     :int  = 1,
                 shortcut  :bool = False,
                 act_type  :str  = "relu",
                 norm_type :str  = "BN",
                 depthwise :bool = False
                 ):
        super(ELANBlock, self).__init__()
        self.inter_dim = out_dim // 2
        self.cv1 = ConvModule(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.Sequential(*[Bottleneck(in_dim    = self.inter_dim,
                                            out_dim   = self.inter_dim,
                                            shortcut  = shortcut,
                                            act_type  = act_type,
                                            norm_type = norm_type,
                                            depthwise = depthwise)
                                            for _ in range(depth)
                                            ])
        self.cv2 = ConvModule((2 + depth) * self.inter_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        # Input proj
        x1, x2 = torch.chunk(self.cv1(x), 2, dim=1)
        out = list([x1, x2])

        # Bottleneck
        out.extend(m(out[-1]) for m in self.m)

        # Output proj
        out = self.cv2(torch.cat(out, dim=1))

        return out
    