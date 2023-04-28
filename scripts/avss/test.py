import torch
import torch.nn
import os
from mmcv import Config
import argparse
from utils.vis_mask import save_color_mask
from utils.compute_color_metrics import calc_color_miou_fscore
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset, get_v2_pallete


def main():
    # logger
    logger = getLogger(None, __name__)
    dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    logger.info(f'Load config from {args.cfg}')

    # config
    cfg = Config.fromfile(args.cfg)
    logger.info(cfg.pretty_text)

    # model
    model = build_model(**cfg.model)
    model.load_state_dict(torch.load(args.weights))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    logger.info('Load trained model %s' % args.weights)

    # Test data
    test_dataset = build_dataset(**cfg.dataset.test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=cfg.dataset.test.batch_size,
                                                  shuffle=False,
                                                  num_workers=cfg.process.num_works,
                                                  pin_memory=True)
    N_CLASSES = test_dataset.num_classes

    # for save predicted rgb masks
    v2_pallete = get_v2_pallete(cfg.dataset.test.label_idx_path)
    resize_pred_mask = cfg.dataset.test.resize_pred_mask
    if resize_pred_mask:
        pred_mask_img_size = cfg.dataset.test.save_pred_mask_img_size
    else:
        pred_mask_img_size = cfg.dataset.test.img_size

    # metrics
    miou_pc = torch.zeros((N_CLASSES))  # miou value per class (total sum)
    Fs_pc = torch.zeros((N_CLASSES))  # f-score per class (total sum)
    cls_pc = torch.zeros((N_CLASSES))  # count per class
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            imgs, audio, mask, vid_temporal_mask_flag, gt_temporal_mask_flag, video_name_list = batch_data
            vid_temporal_mask_flag = vid_temporal_mask_flag.cuda()
            gt_temporal_mask_flag = gt_temporal_mask_flag.cuda()

            imgs = imgs.cuda()
            # audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B * frame, C, H, W)
            mask = mask.view(B * frame, H, W)

            vid_temporal_mask_flag = vid_temporal_mask_flag.view(
                B * frame)  # [B*T]
            gt_temporal_mask_flag = gt_temporal_mask_flag.view(
                B * frame)  # [B*T]

            output, _ = model(audio, imgs, vid_temporal_mask_flag)
            if args.save_pred_mask:
                mask_save_path = os.path.join(
                    args.save_dir, dir_name, 'pred_masks')
                save_color_mask(output, mask_save_path, video_name_list,
                                v2_pallete, resize_pred_mask, pred_mask_img_size, T=10)

            _miou_pc, _fscore_pc, _cls_pc, _ = calc_color_miou_fscore(
                output, mask)
            # compute miou, J-measure
            miou_pc += _miou_pc
            cls_pc += _cls_pc
            # compute f-score, F-measure
            Fs_pc += _fscore_pc

            batch_iou = miou_pc / cls_pc
            batch_iou[torch.isnan(batch_iou)] = 0
            batch_iou = torch.sum(batch_iou) / torch.sum(cls_pc != 0)
            batch_fscore = Fs_pc / cls_pc
            batch_fscore[torch.isnan(batch_fscore)] = 0
            batch_fscore = torch.sum(batch_fscore) / torch.sum(cls_pc != 0)
            logger.info('n_iter: {}, iou: {}, F_score: {}, cls_num: {}'.format(
                n_iter, batch_iou, batch_fscore, torch.sum(cls_pc != 0).item()))

        miou_pc = miou_pc / cls_pc
        logger.info(
            f"[test miou] {torch.sum(torch.isnan(miou_pc)).item()} classes are not predicted in this batch")
        miou_pc[torch.isnan(miou_pc)] = 0
        miou = torch.mean(miou_pc).item()
        miou_noBg = torch.mean(miou_pc[:-1]).item()
        f_score_pc = Fs_pc / cls_pc
        logger.info(
            f"[test fscore] {torch.sum(torch.isnan(f_score_pc)).item()} classes are not predicted in this batch")
        f_score_pc[torch.isnan(f_score_pc)] = 0
        f_score = torch.mean(f_score_pc).item()
        f_score_noBg = torch.mean(f_score_pc[:-1]).item()

        logger.info('test | cls {}, miou: {:.4f}, miou_noBg: {:.4f}, F_score: {:.4f}, F_score_noBg: {:.4f}'.format(
            torch.sum(cls_pc != 0).item(), miou, miou_noBg, f_score, f_score_noBg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='config file path')
    parser.add_argument('weights', type=str, help='model weights path')
    parser.add_argument("--save_pred_mask", action='store_true',
                        default=False, help="save predited masks or not")
    parser.add_argument('--save_dir', type=str,
                        default='work_dir', help='save path')

    args = parser.parse_args()
    main()
