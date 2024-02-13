import torch
import torch.nn as nn
import torch.nn.functional as F


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
    elif act_type == 'gelu':
        return nn.GELU()
    else:
        raise NotImplementedError(act_type)

def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    elif norm_type is None:
        return nn.Identity()
    else:
        raise NotImplementedError(norm_type)


# ----------------- CNN ops -----------------
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


# ----------------- Re-write CNN ops -----------------
class BasicConv(nn.Module):
    def __init__(self, 
                 in_dim,                   # in channels
                 out_dim,                  # out channels 
                 kernel_size=1,            # kernel size 
                 padding=0,                # padding
                 stride=1,                 # padding
                 act_type  :str = 'lrelu', # activation
                 norm_type :str = 'BN',    # normalization
                 depthwise :bool = False
                ):
        super(BasicConv, self).__init__()
        add_bias = False if norm_type else True
        self.depthwise = depthwise
        if not depthwise:
            self.conv = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, g=1, bias=add_bias)
            self.norm = get_norm(norm_type, out_dim)
        else:
            self.conv1 = get_conv2d(in_dim, in_dim, k=kernel_size, p=padding, s=stride, g=1, bias=add_bias)
            self.norm1 = get_norm(norm_type, in_dim)
            self.conv2 = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, g=1, bias=add_bias)
            self.norm2 = get_norm(norm_type, out_dim)
        self.act  = get_activation(act_type)

    def forward(self, x):
        if not self.depthwise:
            return self.act(self.norm(self.conv(x)))
        else:
            # Depthwise conv
            x = self.norm1(self.conv1(x))
            # Pointwise conv
            x = self.norm2(self.conv2(x))
            return x

class RepVggBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='relu', norm_type='BN'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv1 = BasicConv(in_dim, out_dim, kernel_size=3, padding=1, act_type=None, norm_type=norm_type)
        self.conv2 = BasicConv(in_dim, out_dim, kernel_size=1, padding=0, act_type=None, norm_type=norm_type)
        self.act   = get_activation(act_type) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.in_dim, self.out_dim, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: BasicConv):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_blocks=3,
                 expansion=1.0,
                 act_type="silu",
                 norm_type="BN",):
        super(CSPRepLayer, self).__init__()
        inter_dim = int(out_dim * expansion)
        self.conv1 = BasicConv(in_dim, inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.conv2 = BasicConv(in_dim, inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(inter_dim, inter_dim, act_type, norm_type) for _ in range(num_blocks)
        ])
        if inter_dim != out_dim:
            self.conv3 = BasicConv(inter_dim, out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)

        return self.conv3(x_1 + x_2)
