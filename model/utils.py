import os

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from .config import _C as config

def dice_loss(inputs, targets, epoch=None):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """

    inputs = inputs.sigmoid()
    if epoch is not None and epoch >= 8:
        mask = torch.abs(inputs - 0.5) > (0.4 / 50 * epoch)
        inputs = inputs * mask
        targets = targets * mask
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

def sigmoid_focal_loss(inputs, targets, epoch=None, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    prob = inputs.sigmoid()
    if epoch is not None and epoch >= 8:
        mask = torch.abs(prob - 0.5) > (0.4 / 50 * epoch)

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    if epoch is not None and epoch >= 8:
        ce_loss = ce_loss * mask
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()

class ImageTextCorr(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim, out_dim, dropout=0.) -> None:
        super().__init__()

        self.vis_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.unsqueeze = nn.Sequential(
            nn.Conv2d(20, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, l_feat, l_mask):
        B, C, L = l_feat.shape
        vis = self.vis_proj(einops.rearrange(x, 'b c h w -> b h w c'))
        txt = self.text_proj(einops.rearrange(l_feat, 'b c l -> b l c'))
        cost = torch.einsum("bhwc,blc->bhwl", vis, txt) #s=h*w

        cost = einops.rearrange(cost, 'b h w c -> b c h w')
        feat = self.unsqueeze(cost)
        feat = einops.rearrange(feat, 'b c h w -> b h w c')
        out = self.out_proj(feat)
        return out

def create_optimizer(args, model:nn.Module, new_param):
    all_param = model.named_parameters()
    text_backbone = []
    visual_backbone = []
    decoder = []
    others = []
    for n, p in all_param:
        if 'text_encoder' in n:
            text_backbone.append(p)
        elif 'decoder' in n:
            decoder.append(p)
        elif 'backbone' in n and ".".join(n.split(".")[1:]) not in new_param:
            visual_backbone.append(p)
        else:
            others.append(p)
    param = [
        {'params': others},
        {'params': decoder, 'lr': args.lr_decoder},
        {'params': text_backbone, 'lr': args.lr_backbone},
        {'params': visual_backbone, 'lr': args.lr_vssm},
    ]
    optimizer = AdamW(param, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def load_ckpt(backbone, model_size):
    if model_size == 'base':
        name = "vssm_base_0229_ckpt_epoch_237.pth"
        path = f"pretrain/{name}"
        
    st = torch.load(path, map_location='cpu')
    ret = backbone.load_state_dict(st['model'], strict=False)
    ret0_toprint = []
    for k in ret[0]:
        if k.startswith("multimodal_blocks") or k.startswith("text_guidencee") or k.startswith("local_text_fusion"):
            continue
        ret0_toprint.append(k)
    print(ret0_toprint, ret[1])
    return backbone, ret

def update_mamba_config(model_size):
    config_dict = dict(
            patch_size=config.PATCH_SIZE, 
            in_chans=config.IN_CHANS, 
            num_classes=config.NUM_CLASSES, 
            depths=config.DEPTHS, 
            dims=config.EMBED_DIM, 
            # ===================
            ssm_d_state=config.SSM_D_STATE,
            ssm_ratio=config.SSM_RATIO,
            ssm_rank_ratio=config.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.SSM_DT_RANK == "auto" else int(config.SSM_DT_RANK)),
            ssm_act_layer=config.SSM_ACT_LAYER,
            ssm_conv=config.SSM_CONV,
            ssm_conv_bias=config.SSM_CONV_BIAS,
            ssm_drop_rate=config.SSM_DROP_RATE,
            ssm_init=config.SSM_INIT,
            forward_type=config.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MLP_RATIO,
            mlp_act_layer=config.MLP_ACT_LAYER,
            mlp_drop_rate=config.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.DROP_PATH_RATE,
            patch_norm=config.PATCH_NORM,
            norm_layer=config.NORM_LAYER,
            downsample_version=config.DOWNSAMPLE,
            patchembed_version=config.PATCHEMBED,
            gmlp=config.GMLP,
            use_checkpoint=config.USE_CHECKPOINT,
        )

    if model_size == "base":
        config_dict.update(dict(
            patch_size=4,
            depths=[2, 2, 15, 2],
            dims=[128, 128*2, 128*4, 128*8],
            ssm_d_state=1,
            ssm_conv_bias=False,
            mlp_ratio=4.0,
            downsample_version='v3',
            patchembed_version='v2',
            forward_type = "v4noz",
        ))
    return config_dict