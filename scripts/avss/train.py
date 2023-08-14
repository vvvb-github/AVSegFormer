import torch
import time
import torch.nn
import os
import random
import numpy as np
from mmcv import Config
import argparse
from utils import pyutils
from utils.loss_util import LossUtil
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset
from loss import IouSemanticAwareLoss
from utils.compute_color_metrics import calc_color_miou_fscore


def main():
    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # logger
    log_name = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, dir_name)):
        os.mkdir(os.path.join(args.log_dir, dir_name))
    log_file = os.path.join(args.log_dir, dir_name, f'{log_name}.log')
    logger = getLogger(log_file, __name__)
    logger.info(f'Load config from {args.cfg}')

    # config
    cfg = Config.fromfile(args.cfg)
    logger.info(cfg.pretty_text)
    checkpoint_dir = os.path.join(args.checkpoint_dir, dir_name)

    # model
    model = build_model(**cfg.model)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    logger.info("Total params: %.2fM" % (sum(p.numel()
                for p in model.parameters()) / 1e6))

    # dataset
    train_dataset = build_dataset(**cfg.dataset.train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg.dataset.train.batch_size,
                                                   shuffle=True,
                                                   num_workers=cfg.process.num_works,
                                                   pin_memory=True)
    max_step = (len(train_dataset) // cfg.dataset.train.batch_size) * \
        cfg.process.train_epochs
    val_dataset = build_dataset(**cfg.dataset.val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=cfg.dataset.val.batch_size,
                                                 shuffle=False,
                                                 num_workers=cfg.process.num_works,
                                                 pin_memory=True)
    N_CLASSES = train_dataset.num_classes

    # optimizer
    optimizer = pyutils.get_optimizer(model, cfg.optimizer)
    loss_util = LossUtil(**cfg.loss)

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    miou_noBg_list = []
    fscore_list, fscore_noBg_list = [], []
    max_fs, max_fs_noBg = 0, 0
    for epoch in range(cfg.process.train_epochs):
        if epoch == cfg.process.freeze_epochs:
            model.module.freeze_backbone(False)

        for n_iter, batch_data in enumerate(train_dataloader):
            imgs, audio, label, vid_temporal_mask_flag, gt_temporal_mask_flag, _ = batch_data
            vid_temporal_mask_flag = vid_temporal_mask_flag.cuda()
            gt_temporal_mask_flag = gt_temporal_mask_flag.cuda()

            imgs = imgs.cuda()
            # audio = audio.cuda()
            label = label.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B * frame, C, H, W)
            mask_num = 10
            label = label.view(B * mask_num, H, W)
            vid_temporal_mask_flag = vid_temporal_mask_flag.view(
                B * frame)  # [B*T]
            gt_temporal_mask_flag = gt_temporal_mask_flag.view(
                B * frame)  # [B*T]

            # [bs*5, 24, 224, 224]
            output, mask_feature = model(audio, imgs, vid_temporal_mask_flag)
            loss, loss_dict = IouSemanticAwareLoss(
                output, mask_feature, label, gt_temporal_mask_flag, **cfg.loss)
            loss_util.add_loss(loss, loss_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if (global_step - 1) % 20 == 0:
                train_log = 'Iter:%5d/%5d, %slr: %.6f' % (
                            global_step - 1,
                            max_step,
                            loss_util.pretty_out(),
                            optimizer.param_groups[0]['lr'])
                logger.info(train_log)

        # Validation:
        if epoch >= cfg.process.start_eval_epoch and epoch % cfg.process.eval_interval == 0:
            model.eval()

            miou_pc = torch.zeros((N_CLASSES))
            Fs_pc = torch.zeros((N_CLASSES))  # f-score per class (total sum)
            cls_pc = torch.zeros((N_CLASSES))  # count per class
            with torch.no_grad():
                for n_iter, batch_data in enumerate(val_dataloader):
                    # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]
                    imgs, audio, mask, vid_temporal_mask_flag, gt_temporal_mask_flag, _ = batch_data

                    vid_temporal_mask_flag = vid_temporal_mask_flag.cuda()
                    gt_temporal_mask_flag = gt_temporal_mask_flag.cuda()

                    imgs = imgs.cuda()
                    # audio = audio.cuda()
                    mask = mask.cuda()
                    B, frame, C, H, W = imgs.shape
                    imgs = imgs.view(B * frame, C, H, W)
                    mask = mask.view(B * frame, H, W)
                    #! notice
                    vid_temporal_mask_flag = vid_temporal_mask_flag.view(
                        B * frame)  # [B*T]
                    gt_temporal_mask_flag = gt_temporal_mask_flag.view(
                        B * frame)  # [B*T]

                    output, _ = model(audio, imgs, vid_temporal_mask_flag)

                    _miou_pc, _fscore_pc, _cls_pc, _ = calc_color_miou_fscore(
                        output, mask)
                    # compute miou, J-measure
                    miou_pc += _miou_pc
                    cls_pc += _cls_pc
                    # compute f-score, F-measure
                    Fs_pc += _fscore_pc

                miou_pc = miou_pc / cls_pc
                logger.info(
                    f"[miou] {torch.sum(torch.isnan(miou_pc)).item()} classes are not predicted in this batch")
                miou_pc[torch.isnan(miou_pc)] = 0
                miou = torch.mean(miou_pc).item()
                miou_noBg = torch.mean(miou_pc[:-1]).item()
                f_score_pc = Fs_pc / cls_pc
                logger.info(
                    f"[fscore] {torch.sum(torch.isnan(f_score_pc)).item()} classes are not predicted in this batch")
                f_score_pc[torch.isnan(f_score_pc)] = 0
                f_score = torch.mean(f_score_pc).item()
                f_score_noBg = torch.mean(f_score_pc[:-1]).item()

                if miou > max_miou:
                    model_save_path = os.path.join(
                        checkpoint_dir, '%s_miou_best.pth' % (args.session_name))
                    torch.save(model.module.state_dict(), model_save_path)
                    best_epoch = epoch
                    logger.info('save miou best model to %s' % model_save_path)
                if (miou + f_score) > (max_miou + max_fs):
                    model_save_path = os.path.join(
                        checkpoint_dir, '%s_miou_and_fscore_best.pth' % (args.session_name))
                    torch.save(model.module.state_dict(), model_save_path)
                    best_epoch = epoch
                    logger.info('save miou and fscore best model to %s' %
                                model_save_path)

                miou_list.append(miou)
                miou_noBg_list.append(miou_noBg)
                max_miou = max(miou_list)
                max_miou_noBg = max(miou_noBg_list)
                fscore_list.append(f_score)
                fscore_noBg_list.append(f_score_noBg)
                max_fs = max(fscore_list)
                max_fs_noBg = max(fscore_noBg_list)

                val_log = 'Epoch: {}, Miou: {}, maxMiou: {}, Miou(no bg): {}, maxMiou (no bg): {} '.format(
                    epoch, miou, max_miou, miou_noBg, max_miou_noBg)
                val_log += ' Fscore: {}, maxFs: {}, Fscore(no bg): {}, max Fscore (no bg): {}'.format(
                    f_score, max_fs, f_score_noBg, max_fs_noBg)
                logger.info(val_log)

            model.train()

    logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='config file path')
    parser.add_argument('--log_dir', type=str,
                        default='work_dir', help='log dir')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='work_dir', help='dir to save checkpoints')
    parser.add_argument("--session_name", default="AVSS",
                        type=str, help="the AVSS setting")

    args = parser.parse_args()
    main()
