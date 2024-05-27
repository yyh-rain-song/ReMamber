import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizerFast

from .utils import dice_loss, sigmoid_focal_loss


class BaseSegmenter(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.decoder = None
        self.tokenizer = CLIPTokenizerFast.from_pretrained('openai/clip-vit-large-patch14')
        self.text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')

    def forward(self, x, text, mask=None, **kwargs):
        encode_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=20, return_tensors='pt')
        text = encode_text['input_ids'].to(x.device, non_blocking=True)
        l_mask = encode_text['attention_mask'].to(x.device, non_blocking=True)

        input_shape = x.shape[-2:]
        ret = self.text_encoder(text, attention_mask=l_mask)  # (6, 10, 768)
        l_feats = ret['last_hidden_state']
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        if 'pooler_output' in ret:
            pooler_out = ret['pooler_output']
        else:
            pooler_out = None
        
        features = self.backbone(x, l_feats, l_mask, pooler_out=pooler_out)
        x_c1, x_c2, x_c3, x_c4 = features
        pred = self.decoder([x_c4, x_c3, x_c2, x_c1], l_feats, l_mask)
        pred = F.interpolate(pred, input_shape, mode='bilinear', align_corners=True)
        
        # loss
        if self.training:
            loss = dice_loss(pred, mask) + sigmoid_focal_loss(pred, mask, alpha=-1, gamma=0)
            return pred.detach(), mask, loss
        else:
            return pred.detach()