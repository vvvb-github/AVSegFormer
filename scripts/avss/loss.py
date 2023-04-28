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


def IouSemanticAwareLoss(pred_masks, attn_masks, gt_mask, gt_temporal_mask_flag, weight_dict, **kwargs):
    total_loss = 0
    loss_dict = {}

    iou_loss = weight_dict['iou_loss'] * \
        F10_IoU_BCELoss(pred_masks, gt_mask, gt_temporal_mask_flag)
    total_loss += iou_loss
    loss_dict['iou_loss'] = iou_loss.item()

    for i, mask in enumerate(attn_masks):
        loss_i = weight_dict['mask_loss'] * \
            F10_IoU_BCELoss(mask, gt_mask, gt_temporal_mask_flag)
        total_loss += loss_i
        loss_dict[f'mask_loss{i}'] = loss_i.item()

    return total_loss, loss_dict
