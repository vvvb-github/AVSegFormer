import torch
import torch.nn as nn
from .backbone import build_backbone
# from .neck import build_neck
from .head import build_head
from .vggish import VGGish


class AVSegFormer(nn.Module):
    def __init__(self,
                 backbone,
                 vggish,
                 head,
                 neck=None,
                 audio_dim=128,
                 embed_dim=256,
                 T=5,
                 freeze_audio_backbone=True,
                 *args, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.T = T
        self.freeze_audio_backbone = freeze_audio_backbone
        self.backbone = build_backbone(**backbone)
        self.vggish = VGGish(**vggish)
        self.head = build_head(**head)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)

        if self.freeze_audio_backbone:
            for p in self.vggish.parameters():
                p.requires_grad = False
        self.freeze_backbone(True)

        self.neck = neck

    def freeze_backbone(self, freeze=False):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def mul_temporal_mask(self, feats, vid_temporal_mask_flag=None):
        if vid_temporal_mask_flag is None:
            return feats
        else:
            if isinstance(feats, list):
                out = []
                for x in feats:
                    out.append(x * vid_temporal_mask_flag)
            elif isinstance(feats, torch.Tensor):
                out = feats * vid_temporal_mask_flag

            return out

    def extract_feat(self, x):
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats

    def forward(self, audio, frames, targets=None, vid_temporal_mask_flag=None, train=False):
        if vid_temporal_mask_flag is not None:
            vid_temporal_mask_flag = vid_temporal_mask_flag.view(-1, 1, 1, 1)
        with torch.no_grad():
            audio_feat = self.vggish(audio)  # [B*T,128]

        audio_feat = audio_feat.unsqueeze(1)
        audio_feat = self.audio_proj(audio_feat)
        img_feat = self.extract_feat(frames)
        img_feat = self.mul_temporal_mask(img_feat, vid_temporal_mask_flag)

        pred_mask, pred_logit, mask_feature, aux_outputs = self.head(
            img_feat, audio_feat, targets, train=train)

        if vid_temporal_mask_flag is not None:
            pred_mask = self.mul_temporal_mask(
                pred_mask, vid_temporal_mask_flag)
            pred_logit = self.mul_temporal_mask(
                pred_logit, vid_temporal_mask_flag.squeeze(-1))
            mask_feature = self.mul_temporal_mask(
                mask_feature, vid_temporal_mask_flag)
            if aux_outputs is not None:
                for i, (m, l) in enumerate(aux_outputs):
                    aux_outputs[i][0] = self.mul_temporal_mask(
                        m, vid_temporal_mask_flag)
                    aux_outputs[i][1] = self.mul_temporal_mask(
                        l, vid_temporal_mask_flag.squeeze(-1))

        return pred_mask, pred_logit, mask_feature, aux_outputs
