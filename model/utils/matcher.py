import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F


class HungarianMatcher(nn.Module):
    def __init__(self, num_queries, smooth=0.001):
        super().__init__()
        self.num_queries = num_queries
        self.smooth = smooth
        self.l1_loss = nn.L1Loss(reduction='none')

    @torch.no_grad()
    def forward(self, pred_masks, pred_logits, targets):
        """
        pred_masks: [bs, num_queries, h, w]
        pred_logits: [bs, num_queries, num_classes]
        targets: [
            {
                'gt_masks': (num_gts, h, w),
                'gt_classes': (num_gts, num_classes)
                'vid_mask_flag': bool
            } 
        ]=[bs]
        """
        pred_masks = F.interpolate(
            pred_masks, size=targets[0]['gt_masks'].shape[-2:], mode='bilinear', align_corners=False)
        C = []
        # calc dice loss matrix
        for m, lo, tgt in zip(pred_masks, pred_logits, targets):
            if tgt['vid_mask_flag']:
                dice_loss = self.loss_dice(m, tgt)
                logits_loss = self.loss_logits(lo, tgt)
                total_loss = dice_loss+logits_loss
                C.append(total_loss.cpu())
        # matcher
        indices = [linear_sum_assignment(c) for c in C]
        device = pred_masks.device
        return [(torch.as_tensor(i, dtype=torch.int64, device=device),
                 torch.as_tensor(j, dtype=torch.int64, device=device)) for i, j in indices]

    def loss_dice(self, pred_mask, target):
        pred_mask = pred_mask.sigmoid().flatten(-2).unsqueeze(1)
        gt_mask = target['gt_masks'].flatten(-2).unsqueeze(0)
        a = (pred_mask*gt_mask).sum(-1)
        b = pred_mask.sum(-1)
        c = gt_mask.sum(-1)
        loss = -2*a/(b+c+self.smooth)
        return loss

    def loss_logits(self, logits, target):
        logits = logits.sigmoid().unsqueeze(1)
        gt_cls = target['gt_classes'].unsqueeze(0)
        nq, nc = logits.shape[0], gt_cls.shape[1]
        logits = logits.repeat(1, nc, 1)
        gt_cls = gt_cls.repeat(nq, 1, 1)
        loss = self.l1_loss(logits, gt_cls)
        return loss.mean(-1)


def build_matcher(type, **kwargs):
    if type == 'HungarianMatcher':
        return HungarianMatcher(**kwargs)
    else:
        raise ValueError
