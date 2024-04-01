import torch
import torch.nn as nn
import torch.nn.functional as F


def l_loss(pred_logit):
    l = nn.CrossEntropyLoss()

    indices = torch.tensor(list(range(0, len(pred_logit), 5)))
    indices = indices.cuda()
    first_pred = torch.index_select(
        pred_logit, dim=0, index=indices)  # [bs//5, 1, 1]

    loss = l(first_pred, torch.ones_like(first_pred))
    return loss


def dice_loss(pred_mask, first_gt_mask):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs, 1, h, w)
        first_gt_mask (Tensor): (bs//5, h, w)
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)

    indices = torch.tensor(list(range(0, len(pred_mask), 5)))
    indices = indices.cuda()
    first_pred = torch.index_select(
        pred_mask, dim=0, index=indices)  # [bs//5, 1, 224, 224]

    pred_mask = first_pred.flatten(1)
    gt_mask = first_gt_mask.flatten(1)
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


def AVSLoss(pred_mask, pred_logit, mask_feature, aux_outputs, gt_mask, loss_type, weight_dict, **kwargs):
    total_loss = 0
    print_loss_dict = {}

    for l, w in zip(loss_type, weight_dict):
        if l == 'dice':
            loss = w*dice_loss(pred_mask, gt_mask)
            total_loss += loss
            print_loss_dict['dice_loss'] = loss.item()
        elif l == 'l1':
            loss = w*l_loss(pred_logit)
            total_loss += loss
            print_loss_dict['l1_loss'] = loss.item()
        elif l == 'mix':
            loss = w*mix_loss(mask_feature, gt_mask)
            total_loss += loss
            print_loss_dict['mix_loss'] = loss.item()

    if aux_outputs is not None:
        for i, (mask, logit) in enumerate(aux_outputs):
            for l, w in zip(loss_type, weight_dict):
                if l == 'dice':
                    loss = w*dice_loss(mask, gt_mask)
                    total_loss += loss
                    print_loss_dict[f'dice_loss{i}'] = loss.item()
                elif l == 'l1':
                    loss = w*l_loss(logit)
                    total_loss += loss
                    print_loss_dict[f'l1_loss{i}'] = loss.item()

    return total_loss, print_loss_dict
