import torch
import torch.nn
import os
# --- 추가된 모듈 ---
from PIL import Image
import numpy as np
import json 
# --------------------
from mmcv import Config
import argparse
from utils import pyutils
from utility import mask_iou, Eval_Fmeasure, save_mask
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset


def save_mask_as_png(mask_tensor, save_path, threshold=0.5):
    """
    단일 마스크 텐서를 흑백 PNG 파일로 저장합니다.
    """
    # 1. 텐서 정리 및 CPU/NumPy 변환
    mask_tensor = mask_tensor.squeeze() 
    if mask_tensor.is_cuda:
        mask_np = mask_tensor.cpu().numpy()
    else:
        mask_np = mask_tensor.numpy()
        
    # 2. 이진화 및 0-255 스케일링
    mask_np = (mask_np > threshold).astype(np.uint8) * 255
    
    # 3. PIL Image 객체 생성 및 저장
    mask_image = Image.fromarray(mask_np, mode='L')
    
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    mask_image.save(save_path)
    print(f"✅ 마스크가 {save_path}에 저장되었습니다.")


def main():
    # logger
    logger = getLogger(None, __name__)
    dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    logger.info(f'Load config from {args.cfg}')

    # config
    cfg = Config.fromfile(args.cfg)
    
    # 🚨🚨🚨 배치 사이즈를 1로 강제 설정 🚨🚨🚨
    cfg.dataset.test.batch_size = 1 
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
                                                  batch_size=cfg.dataset.test.batch_size, # 이제 항상 1
                                                  shuffle=False,
                                                  num_workers=cfg.process.num_works,
                                                  pin_memory=True)
    
    # 💡 miou 임계값 및 실패 기록 리스트 초기화
    threshold = 0.5
    failed_batches = []
    
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    # 🚨 PNG 저장 기본 경로 설정
    save_png_root = os.path.join(args.save_dir, dir_name, 'wrong_predictions')

    # Test
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            imgs, audio, mask, video_name_list = batch_data

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape # B는 항상 1
            total_frames = B * frame
            
            # 데이터 형태 조정
            imgs = imgs.view(total_frames, C, H, W)
            mask = mask.view(total_frames, H, W)
            audio = audio.view(-1, audio.shape[2],
                               audio.shape[3], audio.shape[4])

            output, _ = model(audio, imgs)
            
            # 성능 지표 계산
            miou = mask_iou(output.squeeze(1), mask)
            current_miou_value = miou.item()
            F_score = Eval_Fmeasure(output.squeeze(1), mask)

            # 💡 기존 save_mask 로직 (필요하면 주석 해제)
            if args.save_pred_mask:
                mask_save_path = os.path.join(
                    args.save_dir, dir_name, 'pred_masks')
                # save_mask(output.squeeze(1), mask_save_path, video_name_list)

            # --- [잘못 예측한 배치 기록 및 PNG 저장 로직] ---
            if current_miou_value < threshold:
                logger.warning(f'🚨 FAILED BATCH {n_iter} (mIoU: {current_miou_value:.4f}). Saving masks...')
                
                # 1. 배치 정보 기록 (B=1이므로 비디오 하나)
                video_name = video_name_list[0]
                failed_batches.append({
                    'iter': n_iter,
                    'miou': current_miou_value,
                    'F_score': F_score, # F_score도 item()으로 실수 변환
                    'video_name': video_name
                })

                # 2. 모든 프레임을 PNG로 저장
                for i in range(total_frames):
                    frame_idx = i + 1
                    
                    # 파일명 구조: [video_name]_frame_[index].png
                    file_name = f'{video_name}_frame_{frame_idx:03d}.png'
                    
                    # 예측 마스크 저장
                    pred_mask_dir = os.path.join(save_png_root, 'pred')
                    pred_mask_path = os.path.join(pred_mask_dir, file_name)
                    save_mask_as_png(output[i], pred_mask_path, threshold=0.5) 
                    
                    # 정답 마스크 저장 (비교용)
                    gt_mask_dir = os.path.join(save_png_root, 'gt')
                    gt_mask_path = os.path.join(gt_mask_dir, file_name)
                    save_mask_as_png(mask[i], gt_mask_path, threshold=0.5)
            # --- [로직 끝] ---

            avg_meter_miou.add({'miou': miou})
            avg_meter_F.add({'F_score': F_score})
            logger.info('n_iter: {}, iou: {:.4f}, F_score: {:.4f}'.format(
                n_iter, current_miou_value, F_score))

        # --- [최종 결과 및 실패 목록 저장] ---
        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        
        logger.info(f'--- Test Finished ---')
        logger.info(f'Total Failed Batches (mIoU < {threshold}): {len(failed_batches)}')
        
        # 실패 배치 목록을 JSON 파일로 저장
        if failed_batches:
            failed_list_path = os.path.join(args.save_dir, dir_name, 'failed_batches.json')
            os.makedirs(os.path.dirname(failed_list_path), exist_ok=True)
            with open(failed_list_path, 'w') as f:
                json.dump(failed_batches, f, indent=4)
            logger.info(f'✅ Failed batches list saved to: {failed_list_path}')

        logger.info(f'test miou: {miou.item():.4f}, F_score: {F_score:.4f}')


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