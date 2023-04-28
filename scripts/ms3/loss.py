import torch
import torch.nn as nn
import torch.nn.functional as F


def F5_IoU_BCELoss(pred_mask, five_gt_masks):
    """
    binary cross entropy loss (iou loss) of the total five frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    five_gt_masks: ground truth mask of the total five frames, shape: [bs*5, 1, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)  # [bs*5, 1, 224, 224]
    # five_gt_masks = five_gt_masks.view(-1, 1, five_gt_masks.shape[-2], five_gt_masks.shape[-1]) # [bs*5, 1, 224, 224]
    loss = nn.BCELoss()(pred_mask, five_gt_masks)

    return loss


def F5_Dice_loss(pred_mask, five_gt_masks):
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
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    return loss.mean()


def IouSemanticAwareLoss(pred_mask, attn_masks, gt_mask, weight_dict, loss_type='bce', **kwargs):
    total_loss = 0
    loss_dict = {}

    if loss_type == 'bce':
        loss_func = F5_IoU_BCELoss
    elif loss_type == 'dice':
        loss_func = F5_Dice_loss
    else:
        raise ValueError

    iou_loss = weight_dict['iou_loss'] * loss_func(pred_mask, gt_mask)
    total_loss += iou_loss
    loss_dict['iou_loss'] = iou_loss.item()

    for i, mask in enumerate(attn_masks):
        loss_i = weight_dict['mask_loss'] * loss_func(mask, gt_mask)
        total_loss += loss_i
        loss_dict[f'mask_loss{i}'] = loss_i.item()

    return total_loss, loss_dict
