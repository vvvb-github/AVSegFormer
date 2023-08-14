import torch
import torch.nn as nn
import torch.nn.functional as F


def F10_IoU_BCELoss(pred_mask, ten_gt_masks, gt_temporal_mask_flag):
    """
    binary cross entropy loss (iou loss) of the total ten frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    ten_gt_masks: ground truth mask of the total ten frames, shape: [bs*10, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    if ten_gt_masks.shape[1] == 1:
        ten_gt_masks = ten_gt_masks.squeeze(1)

    loss = nn.CrossEntropyLoss(reduction='none')(
        pred_mask, ten_gt_masks)  # [bs*10, 224, 224]
    loss = loss.mean(-1).mean(-1)  # [bs*10]
    loss = loss * gt_temporal_mask_flag  # [bs*10]
    loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)

    return loss


def Mix_Dice_loss(pred_mask, norm_gt_mask, gt_temporal_mask_flag):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, h, w)
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)

    pred_mask = pred_mask.flatten(1)
    gt_mask = norm_gt_mask.flatten(1)
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    loss = loss * gt_temporal_mask_flag
    loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)
    return loss


def IouSemanticAwareLoss(pred_masks, mask_feature, gt_mask, gt_temporal_mask_flag, weight_dict, **kwargs):
    total_loss = 0
    loss_dict = {}

    iou_loss = weight_dict['iou_loss'] * \
        F10_IoU_BCELoss(pred_masks, gt_mask, gt_temporal_mask_flag)
    total_loss += iou_loss
    loss_dict['iou_loss'] = iou_loss.item()

    mask_feature = torch.mean(mask_feature, dim=1, keepdim=True)
    mask_feature = F.interpolate(
        mask_feature, gt_mask.shape[-2:], mode='bilinear', align_corners=False)
    one_mask = torch.ones_like(gt_mask)
    norm_gt_mask = torch.where(gt_mask > 0, one_mask, gt_mask)
    mix_loss = weight_dict['mix_loss'] * \
        Mix_Dice_loss(mask_feature, norm_gt_mask, gt_temporal_mask_flag)
    total_loss += mix_loss
    loss_dict['mix_loss'] = mix_loss.item()

    return total_loss, loss_dict
