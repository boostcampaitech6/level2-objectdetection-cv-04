# Import module

import os
import argparse
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from pycocotools.coco import COCO
import pandas as pd
from metrics import meanAveragePrecision

def main(args):
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    # Config file 들고오기
    cfg = Config.fromfile(args.config_dir)
    
    cfg.work_dir = args.output

    if args.train:  # train mode
        # wandb를 사용하기 위한 hook 설정
        cfg.log_config.hooks = [
            dict(type='TextLoggerHook'),
            dict(type='MMDetWandbHook',
                init_kwargs={'project':'level2-object-detection-cv-04',
                                'name':args.wandb_name},
                interval=10,
                log_checkpoint=False,
                log_checkpoint_metadata=True,
                num_eval_images=100)
        ]
        try:
            cfg.data.train.classes = classes
            cfg.data.train.img_prefix = args.root
            cfg.data.train.ann_file = args.root + args.annotation # train json 정보
        except:
            cfg.data.train.dataset.classes = classes
            cfg.data.train.dataset.img_prefix = args.root
            cfg.data.train.dataset.ann_file = args.root + args.annotation # train json 정보

        
        cfg.device = get_device()

        # build_dataset
        datasets = [build_dataset(cfg.data.train)]

        # 모델 build 및 pretrained network 불러오기
        model = build_detector(cfg.model)
        model.init_weights()

        train_detector(model, datasets[0], cfg, distributed=False, validate=False)
    
    else:  # test(inference) mode

        cfg.data.test.classes = classes
        cfg.data.test.img_prefix = args.root
        cfg.data.test.ann_file = args.root + args.annotation # test.json 정보
        cfg.data.test.test_mode = True

        # build dataset & dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
                    dataset,
                    samples_per_gpu=1,
                    workers_per_gpu=cfg.data.workers_per_gpu,
                    dist=False,
                    shuffle=False)
        
        # checkpoint path
        checkpoint_path = os.path.join(cfg.work_dir, f'{args.checkpoint}.pth')
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load
        model.CLASSES = dataset.CLASSES
        model = MMDataParallel(model.cuda(), device_ids=[0])
        output = single_gpu_test(model, data_loader, show_score_thr=0.05) # output 계산
        
        prediction_strings = []
        file_names = []
        coco = COCO(cfg.data.test.ann_file)

        class_num = 10
        for i, out in enumerate(output):
            prediction_string = ''
            image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
            for j in range(class_num):
                for o in out[j]:
                    """
                    기존 out: shape: (5,)
                        x_min, y_min, x_max, y_max, confidence_score

                    -out -> prediction_string 재구성-
                        class_id, confidence_score, x_min, y_min, x_max, y_max
                    """
                    prediction_string += str(j) + ' ' + str(o[4]) + ' ' + \
                                         str(o[0]) + ' ' + str(o[1]) + ' ' + str(o[2]) + ' ' + str(o[3]) + ' '
                
            prediction_strings.append(prediction_string)
            file_names.append(image_info['file_name'])

            submission = pd.DataFrame()
            submission['PredictionString'] = prediction_strings
            submission['image_id'] = file_names

        if args.valid:
            mean_ap, average_precisions = meanAveragePrecision(submission, args)
            print(f"mAP(mean Average Precision): {mean_ap}")
            print(f"average_precisions: {average_precisions}")
        else:
            submission.to_csv(os.path.join(cfg.work_dir, f'submission_{args.checkpoint}.csv'), index=None)
            submission.head()
'''
1. train_fold_1을 학습 시킨다.
2. val_fold 1으로 mAP 확인

1~2를 각 fold별로 반복
--> 모델 10개
모델 10개를 모두 test 데이터셋으로 inference

3. train_fold_1으로 학습 시킨 모델로 test 데이터셋 inference(submission)
4. 리더보드 제출 후 test의 mAP와, Validation set의 mAP 비교. (터무니 없이 차이가 나는지?)
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 데이터셋 위치
    parser.add_argument(
        "--root",
        type=str,
        default="../../dataset/",
        help="dataset's location (default: ../../dataset/)"
    )
    # config 파일 정보
    parser.add_argument(
        "--annotation",
        type=str,
        default="train.json",
        help="annotation file name (default: train.json)"
    )
    # output 위치
    parser.add_argument(
        "--output",
        type=str,
        default="./work_dirs/default",
        help="output's location (default: ./work_dirs/default)"
    )
    
    # train/test mode
    parser.add_argument(
        "--train",
        type=int,
        default=1,
        help="set inference/train mode, 0: inference / 1: train"
    )
    # valid/submission mode
    parser.add_argument(
        "--valid",
        type=int,
        default=1,
        help="set submission/valid mode, 0: submission / 1: valid"
    )
    # Config file
    parser.add_argument(
        "--config_dir", 
        type=str,
        default=None,
        help="config file's location",
    )
    # wandb name
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help='name of wandb test name'
    )

    # model pth name
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="name of checkpoint want to inference"
    )
    
    # ground truth name
    parser.add_argument(
        "--gt_name",
        type=str,
        default=None,
        help="name of Ground Truth name"
    )
    args = parser.parse_args()   

    if args.config_dir == None:
        raise Exception("Import config file's location")
    
    if args.train and args.wandb_name == None:
        raise Exception("Import wandb test name")
    
    if args.output == "./work_dirs/default":
        print("Warning: Your output directory is set to (./work_dirs/default), you should change your output directory.")
    
    if args.checkpoint == "latest":
        print("Warning: Your model name is set to (latest). If not intended, change your model name.")

    if not args.train and args.valid and args.gt_name == None:
        raise Exception("Import GT name")
    
    print(args)  # 내가 설정한 augrments
    main(args)