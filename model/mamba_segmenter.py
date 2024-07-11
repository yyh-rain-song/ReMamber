
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model

from .base_segmenter import BaseSegmenter
from .customized_model import (ReMamber, ImageTextCorr, LayerNorm2d,
                               Linear2d, VSSLayer)
from .utils import Fusion as FuseLayer
from .utils import conv_layer, load_ckpt, update_mamba_config


class UpSample2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        # self.dim = dim*2
        if not channel_first:
            raise
        self.proj = Linear2d(dim, dim//2, bias=False)
        self.norm = norm_layer(dim//2)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.proj(x)
        x= self.norm(x)
        return x

class MambaDecoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        # self.hidden_dim = dims[0]*2
        self.channel_first = True
        depths = [2,4,2,2]
        self.num_layers = len(depths)
        dims = kwargs['dims'][0]
        dims = [dims*8, dims*4, dims*2, dims]
        use_checkpoint = kwargs['use_checkpoint']
        norm_layer = kwargs['norm_layer']

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)

        self.text_guidencee = nn.ModuleList()
        self.local_text_fusion = nn.ModuleList()
        self.multimodal_blocks = nn.ModuleList()
        self.in_proj = nn.ModuleList()
        self.hire_fusion = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim = dims[i_layer],
                depth = depths[i_layer],
                # drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                ssm_act_layer=ssm_act_layer,
                downsample=UpSample2D,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=kwargs['ssm_d_state'],
                ssm_ratio=kwargs['ssm_ratio'],
                ssm_dt_rank=kwargs['ssm_dt_rank'],
                # ssm_act_layer=kwargs['ssm_act_layer'],
                ssm_conv=kwargs['ssm_conv'],
                ssm_conv_bias=kwargs['ssm_conv_bias'],
                ssm_drop_rate=kwargs['ssm_drop_rate'],
                ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'],
                # =================
                mlp_ratio=kwargs['mlp_ratio'],
                mlp_act_layer=kwargs['mlp_act_layer'],
                mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'],
                forward_coremm='SS2D',
            )
            self.multimodal_blocks.append(layer)
            self.in_proj.append(Linear2d(3*dims[i_layer], dims[i_layer], bias=False))


            self.text_guidencee.append(
                nn.Sequential(
                    nn.Linear(768, dims[i_layer]),
                    nn.ReLU(),
                )
            )

            self.local_text_fusion.append(
                ImageTextCorr(
                    visual_dim=dims[i_layer],
                    text_dim=768,
                    hidden_dim=512,
                    out_dim=dims[i_layer],
                )
            )
            if i_layer != self.num_layers - 1:
                self.hire_fusion.append(
                    FuseLayer(
                        dims[i_layer]//2, dims[i_layer]//2, dims[i_layer]//2
                    )
                )

        self.proj_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(dims[3]//2, dims[3]//2, 3, padding=1),
            nn.Conv2d(dims[3]//2, 1, 3, padding=1),
        )


    
    def forward(self, x, l_feat, l_mask, pooler_out=None):

        feat = x[0]
        for i,layer in enumerate(self.multimodal_blocks):
            _, c, h, w = feat.shape

            if pooler_out is None:
                pooling_text = l_feat[..., 0]
            else:
                pooling_text = pooler_out
            text_guidence = self.text_guidencee[i](pooling_text)
            text_guidence = einops.repeat(text_guidence, "b c -> b c h w", h=h, w=w)
            local_text = self.local_text_fusion[i](feat, l_feat, l_mask)
            local_text = einops.rearrange(local_text, 'b h w c -> b c h w', h=h)
            mm_input = torch.cat([feat, text_guidence, local_text], dim=1)
            
            feat, _ = layer(self.in_proj[i](mm_input), None, None)
            if i+1 < len(x):
                feat = self.hire_fusion[i](feat, x[i+1])
                feat = feat + x[i+1]

        out = self.proj_out(feat)
        return out


class MambaSegmentor(BaseSegmenter):
    def __init__(self, backbone, **kwargs):
        super().__init__(backbone)
        self.decoder = MambaDecoder(**kwargs)



@register_model
def ReMamber_Mamba(img_size=256, model_size="tiny", **kwargs):
    config_dict = update_mamba_config(model_size)
    backbone = ReMamber(img_size=img_size, **config_dict)
    backbone, ret = load_ckpt(backbone, model_size)
    return MambaSegmentor(backbone, **config_dict), ret[0]
