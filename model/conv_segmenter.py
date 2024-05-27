import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model

from .base_segmenter import BaseSegmenter
from .customized_model import ReMamber
from .utils import load_ckpt, update_mamba_config


class Conv2d_BN(nn.Module):
    """Convolution with BN module."""
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)
        self.bn = norm_layer(out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        """foward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x


class ResBlock(nn.Module):
    """Residual block for convolutional local feature."""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.Hardswish,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = Conv2d_BN(in_features,
                               hidden_features,
                               act_layer=act_layer)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        self.norm = norm_layer(hidden_features)
        self.act = act_layer()
        self.conv2 = Conv2d_BN(hidden_features, out_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        initialization
        """
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        """foward function"""
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)
        return feat
        # return identity + feat

class ConvDecoder(nn.Module):
    def __init__(self, in_dim, **kwargs) -> None:
        super().__init__()
        self.in_dim = [in_dim*8, in_dim*4, in_dim*2, in_dim]
        self.hidden_dim = in_dim*4

        self.mixer3 = ResBlock(in_dim*8, out_features=self.hidden_dim)

        self.mixer2 = ResBlock(in_dim*4+self.hidden_dim, out_features=self.hidden_dim)

        self.mixer1 = ResBlock(in_dim*2+self.hidden_dim, out_features=self.hidden_dim)

        self.neck = nn.Sequential(
            nn.Conv2d(self.hidden_dim+in_dim, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.hidden_dim, 1, 3, padding=1)
        )
    
    def forward(self, x, *args):

        out0, out1, out2, out3 = x
        out = self.mixer3(out0)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        
        out = self.mixer2(torch.cat([out, out1], dim=1))
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        
        out = self.mixer1(torch.cat([out, out2], dim=1))
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        out = self.neck(torch.cat([out, out3], dim=1))

        return out

class ConvSegmentor(BaseSegmenter):
    def __init__(self, backbone, img_size=256, patch_size=4, embed_dim=256, **kwargs):
        super().__init__(backbone)
        res = img_size // patch_size
        self.decoder = ConvDecoder(in_dim=embed_dim, resolution=res)



@register_model
def ReMamber_Conv(img_size, model_size="base", **kwargs):
    config_dict = update_mamba_config(model_size)
    backbone = ReMamber(**config_dict)
    backbone, ret = load_ckpt(backbone, model_size)
    model = ConvSegmentor(backbone, img_size=img_size, embed_dim=config_dict['dims'][0], patch_size=config_dict['patch_size'])
    return model, ret[0]

