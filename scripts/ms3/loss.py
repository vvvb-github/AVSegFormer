import torch
import torch.nn as nn
import torch.nn.functional as F


def l1_loss(pred_logit):
    l1 = nn.L1Loss()
    pred_logit = pred_logit.sigmoid()
    loss = l1(pred_logit, torch.ones_like(pred_logit))
    return loss


def dice_loss(pred_mask, five_gt_masks):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, h, w)
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)

    pred_mask = pred_mask.flatten(1)
    gt_mask = five_gt_masks.flatten(1)
    a = (pred_mask * gt_mask).sum(-1)
    b = pred_mask.sum(-1)
    c = gt_mask.sum(-1)
    d = (2 * a) / (b + c + 0.001)
    loss = 1 - d
    return loss.mean()


def mix_loss(mask_feature, gt_mask):
    mask_feature = torch.mean(mask_feature, dim=1, keepdim=True)
    mask_feature = F.interpolate(
        mask_feature, gt_mask.shape[-2:], mode='bilinear', align_corners=False)
    return dice_loss(mask_feature, gt_mask)


def AVSLoss(pred_mask, pred_logit, mask_feature, gt_mask, loss_type, weight_dict, **kwargs):
    total_loss = 0
    print_loss_dict = {}

    for l, w in zip(loss_type, weight_dict):
        if l == 'dice':
            loss = w*dice_loss(pred_mask, gt_mask)
            total_loss += loss
            print_loss_dict['dice_loss'] = loss.item()
        elif l == 'l1':
            loss = w*l1_loss(pred_logit)
            total_loss += loss
            print_loss_dict['f1_loss'] = loss.item()
        elif l == 'mix':
            loss = w*mix_loss(mask_feature, gt_mask)
            total_loss += loss
            print_loss_dict['mix_loss'] = loss.item()

    return total_loss, print_loss_dict
