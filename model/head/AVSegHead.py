import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import build_transformer, build_positional_encoding, build_fusion_block, build_generator, build_matcher
from ops.modules import MSDeformAttn
from torch.nn.init import normal_
from torch.nn.functional import interpolate


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()

        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Conv2d(n, k, kernel_size=1, stride=1, padding=0)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SimpleFPN(nn.Module):
    def __init__(self, channel=256, layers=3):
        super().__init__()

        assert layers == 3  # only designed for 3 layers
        self.up1 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        )
        self.up2 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        )
        self.up3 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x1 = self.up1(x[-1])
        x1 = x1 + x[-2]

        x2 = self.up2(x1)
        x2 = x2 + x[-3]

        y = self.up3(x2)
        return y


class AVSegHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 query_num,
                 transformer,
                 query_generator,
                 matcher,
                 embed_dim=256,
                 valid_indices=[1, 2, 3],
                 scale_factor=4,
                 positional_encoding=None,
                 use_learnable_queries=True,
                 fusion_block=None) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_num = query_num
        self.valid_indices = valid_indices
        self.num_feats = len(valid_indices)
        self.scale_factor = scale_factor
        self.use_learnable_queries = use_learnable_queries
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feats, embed_dim))
        self.learnable_query = nn.Embedding(query_num, embed_dim)

        self.query_generator = build_generator(**query_generator)
        self.matcher = build_matcher(**matcher)

        self.transformer = build_transformer(**transformer)
        if positional_encoding is not None:
            self.positional_encoding = build_positional_encoding(
                **positional_encoding)
        else:
            self.positional_encoding = None

        in_proj = []
        for c in in_channels:
            in_proj.append(
                nn.Sequential(
                    nn.Conv2d(c, embed_dim, kernel_size=1),
                    nn.GroupNorm(32, embed_dim)
                )
            )
        self.in_proj = nn.ModuleList(in_proj)

        if fusion_block is not None:
            self.fusion_block = build_fusion_block(**fusion_block)

        self.lateral_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, embed_dim)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, embed_dim),
            nn.ReLU(True)
        )

        self.fpn = SimpleFPN()
        self.logits_predictor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Linear(128, 1)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def reform_output_squences(self, memory, spatial_shapes, level_start_index, dim=1):
        split_size_or_sections = [None] * self.num_feats
        for i in range(self.num_feats):
            if i < self.num_feats - 1:
                split_size_or_sections[i] = level_start_index[i +
                                                              1] - level_start_index[i]
            else:
                split_size_or_sections[i] = memory.shape[dim] - \
                    level_start_index[i]
        y = torch.split(memory, split_size_or_sections, dim=dim)
        return y

    def forward(self, feats, audio_feat, targets, train):
        """
        Args:
            feats (list(tensor)): [(bs, c, h, w)]=[4]
            audio_feat (tensor): tensor: (bs, 1, c)
            targets (dict): [
                {
                    'gt_masks': (num_gts, h, w),
                    'gt_classes': (num_gts, num_classes)
                    'vid_mask_flag': bool
                } 
            ]=[bs]
            train (bool)
        """
        feat14 = self.in_proj[0](feats[0])
        srcs = [self.in_proj[i](feats[i]) for i in self.valid_indices]
        masks = [torch.zeros((x.size(0), x.size(2), x.size(
            3)), device=x.device, dtype=torch.bool) for x in srcs]
        pos_embeds = []
        for m in masks:
            pos_embeds.append(self.positional_encoding(m))
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # prepare queries
        bs = audio_feat.shape[0]
        query = self.query_generator(audio_feat)
        if self.use_learnable_queries:
            query = query + \
                self.learnable_query.weight[None, :, :].repeat(bs, 1, 1)

        memory, outputs = self.transformer(query, src_flatten, spatial_shapes,
                                           level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # generate mask feature
        mask_feats = []
        for i, z in enumerate(self.reform_output_squences(memory, spatial_shapes, level_start_index, 1)):
            mask_feats.append(z.transpose(1, 2).view(
                bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
        cur_fpn = self.lateral_conv(feat14)
        mask_feature = mask_feats[0]
        mask_feature = cur_fpn + \
            F.interpolate(
                mask_feature, size=cur_fpn.shape[-2:], mode='bilinear', align_corners=False)
        mask_feature = self.out_conv(mask_feature)
        if hasattr(self, 'fusion_block'):
            mask_feature = self.fusion_block(mask_feature, audio_feat)

        # predict output mask
        pred_masks = torch.einsum(
            'bqc,bchw->bqhw', outputs[-1], mask_feature)
        pred_logits = self.logits_predictor(outputs[-1])
        pred_mask = []
        pred_logit = []

        if train:
            # matcher
            indices = self.matcher(pred_masks, pred_logits, targets)
            idx = 0
            for b, tgt in enumerate(targets):
                if tgt['vid_mask_flag']:
                    pm, pl = self.pad_pred_masks(
                        pred_masks[b], pred_logits[b], indices[idx], tgt)
                    idx += 1
                    pred_mask.append(pm)
                    pred_logit.append(pl)
                else:
                    pred_mask.append(torch.zeros([1, self.num_classes, pred_masks.shape[-2],
                                     pred_masks.shape[-1]], dtype=pred_masks.dtype, device=pred_masks.device))
                    pred_logit.append(torch.zeros(
                        [1, self.num_classes, self.num_classes], dtype=pred_logits.dtype, device=pred_logits.device))
        else:
            for m, l in zip(pred_masks, pred_logits):
                pm, pl = self.generate_outputs(m, l)
                pred_mask.append(pm)
                pred_logit.append(pl)

        pred_mask = torch.cat(pred_mask, dim=0)
        pred_logit = torch.cat(pred_logit, dim=0)
        return pred_mask, pred_logit, mask_feature  # (bs, n_cls, h, w), (bs, n_cls, n_cls), (bs, c, h, w)

    def pad_pred_masks(self, pred_mask, pred_logit, indice, target):
        matched_masks = pred_mask[indice[0], :, :]  # (n_cls, h, w)
        matched_logits = pred_logit[indice[0], :]  # (n_cls, n_cls)
        gt_logits = target['gt_classes'][indice[1], :]  # (n_cls, n_cls)
        matched_cls = torch.argmax(gt_logits, dim=1)
        pad_mask = torch.zeros_like(pred_mask[0]).unsqueeze(
            0).repeat(self.num_classes, 1, 1)
        pad_logit = torch.zeros_like(pred_logit[0]).unsqueeze(
            0).repeat(self.num_classes, 1)
        for m, l, cls in zip(matched_masks, matched_logits, matched_cls):
            pad_mask[cls] = m
            pad_logit[cls] = l
        return pad_mask.unsqueeze(0), pad_logit.unsqueeze(0)

    def generate_outputs(self, pred_mask, pred_logit):
        indices = torch.argmax(pred_logit, dim=0)  # (,n_cls)
        vid_logit = pred_logit[indices, :]
        vid_mask = pred_mask[indices, :, :]
        return vid_mask.unsqueeze(0), vid_logit.unsqueeze(0)
