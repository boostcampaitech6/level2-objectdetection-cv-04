# Import module

import argparse
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device


def main(args):
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    # Config file 들고오기
    cfg = Config.fromfile(args.config_dir)
    
    # wandb를 사용하기 위한 hook 설정
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
            init_kwargs={'project':'level2-object-detection-cv-04',
                            'name':args.wandb_name},
            interval=10,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=100)
    ]
    
    if args.train:  # train mode
        try:
            cfg.data.train.classes = classes
            cfg.data.train.img_prefix = args.root
            cfg.data.train.ann_file = args.root + args.annotation # train json 정보
        except:
            cfg.data.train.dataset.classes = classes
            cfg.data.train.dataset.img_prefix = args.root
            cfg.data.train.dataset.ann_file = args.root + args.annotation # train json 정보

        cfg.work_dir = args.output
        cfg.device = get_device()

        # build_dataset
        datasets = [build_dataset(cfg.data.train)]

        # 모델 build 및 pretrained network 불러오기
        model = build_detector(cfg.model)
        model.init_weights()

        train_detector(model, datasets[0], cfg, distributed=False, validate=False)
    else:  # test(inference) mode
        pass

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
        help="dataset's location (default: ../../dataset/)"
    )
    # train/test mode
    parser.add_argument(
        "--train",
        type=int,
        default=1,
        help="set train/test mode, 0: test / 1: train"
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
    
    args = parser.parse_args()
    
    if args.config_dir == None:
        raise Exception("Import config file's location")
    
    if args.wandb_name == None:
        raise Exception("Import wandb test name")
    
    if args.output == "./work_dirs/default":
        print("Warning: Your output directory is set to (./work_dirs/default), you should change your output directory.")
    
    print(args)  # 내가 설정한 augrments
    main(args)

    