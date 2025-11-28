import torch
import torch.nn
import os
from PIL import Image
import numpy as np
from mmcv import Config
import argparse
from utils import pyutils
from utility import mask_iou, Eval_Fmeasure, save_mask
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset
import json # 💡 JSON 저장을 위해 추가
def save_mask_as_png(mask_tensor, save_path, threshold=0.5):
    """
    단일 예측 마스크 텐서를 흑백 PNG 파일로 저장합니다.
    
    Args:
        mask_tensor (torch.Tensor): 저장할 단일 마스크 텐서 (예: [H, W] 또는 [1, H, W] 형태).
        save_path (str): 파일을 저장할 전체 경로 (예: './wrong_preds/pred_001.png').
        threshold (float): 텐서를 이진화(0 또는 255)할 임계값.
    """
    
    # 1. 텐서 정리 및 CPU/NumPy 변환
    # [H, W] 형태로 차원 정리 (squeeze)
    mask_tensor = mask_tensor.squeeze() 
    
    # GPU에서 CPU로 이동 후 NumPy 배열로 변환
    if mask_tensor.is_cuda:
        mask_np = mask_tensor.cpu().numpy()
    else:
        mask_np = mask_tensor.numpy()
        
    # 2. 이진화 및 0-255 스케일링
    # 마스크가 0~1 사이의 값이라고 가정하고 이진화합니다. (threshold를 기준으로 0 또는 255)
    mask_np = (mask_np > threshold).astype(np.uint8) * 255
    
    # 3. PIL Image 객체 생성 및 저장
    # 'L' 모드는 8비트 그레이스케일(흑백) 이미지에 적합합니다.
    mask_image = Image.fromarray(mask_np, mode='L')
    
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    mask_image.save(save_path)
    print(f"✅ 마스크가 {save_path}에 저장되었습니다.")


# *주의*: 마스크가 다중 채널(Multi-class)인 경우 (AVSS는 2차원 마스크가 아닐 수 있음),
# 이진화 대신 각 채널별로 저장하거나 컬러 맵을 적용해야 합니다.
# 여기서는 가장 기본적인 단일 흑백 마스크로 가정합니다.
def main():
    # logger
    logger = getLogger(None, __name__)
    dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    logger.info(f'Load config from {args.cfg}')

    # config
    cfg = Config.fromfile(args.cfg)
    
    # 💡 배치 크기를 1로 강제 설정하여 개별 비디오 분석을 용이하게 함
    # 이 라인을 추가하지 않을 경우, args.cfg 파일에서 batch_size를 1로 변경해야 합니다.
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
                                                  batch_size=cfg.dataset.test.batch_size, # 1
                                                  shuffle=False,
                                                  num_workers=cfg.process.num_works,
                                                  pin_memory=True)
    
    # 💡 잘못 예측한 비디오를 기록할 리스트 초기화
    failed_videos = [] 
    threshold = 0.5  # miou 임계값 설정
    
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    # Test
    logger.info(f'Starting test with batch_size={cfg.dataset.test.batch_size}')
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
            imgs, audio, mask, category_list, video_name_list = batch_data

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            
            # 💡 B는 1이므로, total_frames는 frame 개수와 같습니다.
            total_frames = B * frame 
            
            # 데이터 형태 조정
            imgs = imgs.view(total_frames, C, H, W)
            mask = mask.view(total_frames, H, W)
            audio = audio.view(-1, audio.shape[2],
                               audio.shape[3], audio.shape[4])

            output, _ = model(audio, imgs)
            
            # 성능 지표 계산
            # miou는 모든 프레임의 평균 miou 단일 값(0-dim Tensor)을 반환한다고 가정
            miou = mask_iou(output.squeeze(1), mask) 
            F_score = Eval_Fmeasure(output.squeeze(1), mask)

            # 💡 예측 마스크 저장 (오류 분석을 위해 필수)
            if args.save_pred_mask:
                mask_save_path = os.path.join(
                    args.save_dir, dir_name, 'pred_masks')
                save_mask(output.squeeze(1), mask_save_path,
                          category_list, video_name_list)

            # --- [잘못 예측한 비디오 골라내기 로직] ---
            # batch_size가 1이므로, miou 값은 이 비디오의 전체 평균입니다.
            current_miou_value = miou.item()
            save_png_root = os.path.join(args.save_dir, dir_name, 'wrong_predictions')
            
            if current_miou_value < threshold:
                video_name = video_name_list[0]
                category = category_list[0]

                logger.warning(f'🚨 FAILED! Saving masks for {video_name} (mIoU: {current_miou_value:.4f})')

                for i in range(total_frames):
                    frame_idx = i + 1 # 프레임 인덱스 (1부터 시작)
                    # 예측 마스크 저장 경로
                    pred_mask_dir = os.path.join(save_png_root, category, video_name, 'pred')
                    pred_mask_path = os.path.join(pred_mask_dir, f'frame_{frame_idx:03d}.png')
                    # output[i]는 i번째 프레임의 예측 마스크입니다.
                    save_mask_as_png(output[i], pred_mask_path, threshold=0.5) 

                    # 정답 마스크 저장 경로 (비교용)
                    gt_mask_dir = os.path.join(save_png_root, category, video_name, 'gt')
                    gt_mask_path = os.path.join(gt_mask_dir, f'frame_{frame_idx:03d}.png')
                    # mask[i]는 i번째 프레임의 정답 마스크입니다.
                    save_mask_as_png(mask[i], gt_mask_path, threshold=0.5)
                failed_videos.append({
                    'video_name': video_name,
                    'category': category,
                    'miou': current_miou_value,
                    'n_iter': n_iter 
                })
                logger.warning(f'🚨 FAILED: {video_name} (Category: {category}, mIoU: {current_miou_value:.4f})')
            # --- [로직 끝] ---

            avg_meter_miou.add({'miou': miou})
            avg_meter_F.add({'F_score': F_score})
            logger.info('n_iter: {}, iou: {:.4f}, F_score: {:.4f}'.format(
                n_iter, current_miou_value, F_score))

        # --- [최종 결과 및 실패 목록 저장] ---
        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        
        logger.info(f'--- Test Finished ---')
        logger.info(f'Total Failed Videos (mIoU < {threshold}): {len(failed_videos)}')
        logger.info(f'Test miou: {miou.item():.4f}, F_score: {F_score:.4f}')
        
        # 실패 비디오 목록을 JSON 파일로 저장
        if failed_videos:
            failed_list_path = os.path.join(args.save_dir, dir_name, 'failed_videos.json')
            os.makedirs(os.path.dirname(failed_list_path), exist_ok=True)
            with open(failed_list_path, 'w') as f:
                json.dump(failed_videos, f, indent=4)
            logger.info(f'✅ Failed videos list saved to: {failed_list_path}')


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