import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

import einops

from vmamba_model.vmamba import SS2D, VSSM, LayerNorm2d, Linear2d

from .utils import ImageTextCorr


class Twister(nn.Module):
    def __init__(
            self,         
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3, # < 2 means no conv 
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            input_res=32,
            **kwargs,
        ):
        super().__init__()
        self.input_res = input_res
        self.ss2d = SS2D(d_model, d_state, ssm_ratio, dt_rank, act_layer, d_conv, conv_bias, dropout, bias, dt_min, dt_max, dt_init, dt_scale, dt_init_floor, initialize, forward_type, channel_first, **kwargs)
        self.ss2d.in_proj = Linear2d(d_model * 3, self.ss2d.in_proj.weight.shape[0], bias=bias)
        self.input_res = min(input_res, 16)
        forward_type_1d = "v052d"
        self.ss1d = SS2D(self.input_res**2, d_state, ssm_ratio, dt_rank, act_layer, d_conv, conv_bias, dropout, bias, dt_min, dt_max, dt_init, dt_scale, dt_init_floor, initialize, forward_type_1d, channel_first, **kwargs)
    
    def forward(self, x: torch.Tensor, **kwargs):
        img, global_cond, local_cond = x
        B, C, H, W = img.shape
        if global_cond is not None:
            x = torch.cat([img, global_cond, local_cond], dim=1) # b l 3c
        else:
            x = torch.cat([img, local_cond], dim=-1)
        x_prepaired = x
        x_mix = F.interpolate(x, size=(self.input_res,self.input_res), mode='bilinear')
        x_mix = x_mix.view(B, -1, self.input_res**2)
        x_mix = x_mix.permute(0, 2, 1).unsqueeze(-2).contiguous() # b, hw, 1, c
        x_mix = self.ss1d(x_mix)
        x_mix = x_mix.squeeze(-2).permute(0, 2, 1).contiguous() # b, c, hw
        x_mix = F.interpolate(x_mix.view(B,-1,self.input_res,self.input_res), size=(H,W), mode='bilinear')

        x = x_mix + x_prepaired
        
        out = self.ss2d(x)

        out = [out, global_cond, local_cond]
        return out

class VSSBlock(nn.Module):
    def __init__(
        self,
        forward_coremm='SS2D',
        **kwargs,
    ):
        super().__init__()
        norm_layer = kwargs['norm_layer']
        dim = kwargs['dim']
        drop_path = kwargs['drop_path']
        self.ln_1 = norm_layer(dim)
        self.forward_coremm = forward_coremm
        if not forward_coremm:
            raise
        elif forward_coremm == 'SS2D':
            self.self_attention = SS2D(
                d_model=dim,
                d_state=kwargs['ssm_d_state'],
                dt_rank=kwargs['ssm_dt_rank'],
                act_layer=kwargs['ssm_act_layer'],
                d_conv=kwargs['ssm_conv'],
                conv_bias=kwargs['ssm_conv_bias'],
                dropout=kwargs['ssm_drop_rate'],
                initialize=kwargs['ssm_init'],
                **kwargs,
            )
        elif forward_coremm == 'Twister':
            self.self_attention = Twister(
                d_model=dim,
                d_state=kwargs['ssm_d_state'],
                ssm_ratio=kwargs['ssm_ratio'],
                dt_rank=kwargs['ssm_dt_rank'],
                act_layer=kwargs['ssm_act_layer'],
                d_conv=kwargs['ssm_conv'],
                conv_bias=kwargs['ssm_conv_bias'],
                dropout=kwargs['ssm_drop_rate'],
                initialize=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'],
                channel_first=kwargs['channel_first'],
            )
        else:
            raise
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        if isinstance(input, torch.Tensor):
            out = self.ln_1(input)
            out = self.self_attention(out)
            out = input + self.drop_path(out)
            x = out
        else:
            # input should be a list (img and global / local conditions)
            out = [self.ln_1(i) if i is not None else None for i in input]
            out = self.self_attention(out)
            out = [i + self.drop_path(o) if i is not None else None for i, o in zip(input, out)]
            x = out
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        depth,
        downsample=None,
        forward_coremm="Twister",
        **kwargs,
    ):
        super().__init__()
        # dim = kwargs['dim']
        drop_path = 0.
        dim = kwargs['dim']
        use_checkpoint = kwargs['use_checkpoint']
        norm_layer = kwargs['norm_layer']


        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                forward_coremm=forward_coremm,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                **kwargs,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, channel_first=kwargs['channel_first'])
        else:
            self.downsample = None

    def forward(self, x, l_feat, l_mask):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # x: b w h c
        inner = x
        if self.downsample is not None:
            x = self.downsample(x)

        return x, inner



class ReMamber(VSSM):
    """still extract feature"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = None
        dims = kwargs['dims']
        use_checkpoint = kwargs['use_checkpoint']
        norm_layer = kwargs['norm_layer']
        self.text_guidencee = nn.ModuleList()
        self.local_text_fusion = nn.ModuleList()
        
        self.multimodal_blocks = nn.ModuleList()
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


        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim = self.dims[i_layer],
                depth = 2,
                # drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                ssm_act_layer=ssm_act_layer,
                downsample=None,
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

                forward_coremm='Twister',
            )
            self.multimodal_blocks.append(layer)

            guidence = nn.Sequential(
                nn.Linear(768, dims[i_layer]),
                nn.ReLU(),
            )
            self.text_guidencee.append(guidence)

            fusing = ImageTextCorr(
                visual_dim=dims[i_layer],
                text_dim=768,
                hidden_dim=512,
                out_dim=dims[i_layer],
            )
            self.local_text_fusion.append(fusing)

    def forward_layer(self, x, layer):
        inner = layer.blocks(x)
        out = layer.downsample(inner)
        return out, inner
    
    def forward(self, x, l_feat, l_mask, pooler_out=None):
        x = self.patch_embed(x)
        outs = []
        
        for i,layer in enumerate(self.layers):
            x, inner = self.forward_layer(x, layer)
            _, c, h, w = inner.shape
            out = inner
            if pooler_out is None:
                pooling_text = l_feat[..., 0]
            else:
                pooling_text = pooler_out
            text_guidence = self.text_guidencee[i](pooling_text)
            text_guidence = einops.repeat(text_guidence, "b c -> b c h w", h=h, w=w)
            local_text = self.local_text_fusion[i](out, l_feat, l_mask)
            local_text = einops.rearrange(local_text, 'b h w c -> b c h w', h=h)
            
            mm_input = (out, text_guidence, local_text)
            ret = self.multimodal_blocks[i](mm_input, None, None)[0]
            img_feat = ret[0]
            if layer.downsample is not None:
                x = layer.downsample(img_feat)
            else:
                x = img_feat

            outs.append(img_feat)

        return outs

