# Commons
import os
import argparse
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo

# Test
from mmdet.engine.hooks.utils import trigger_visualization_hook


def main(args):
    # Reduce the number of repeated compilations and improve
    # training/testing speed.
    setup_cache_size_limit_of_dynamo()
    
    # config file 불러오기
    cfg = Config.fromfile(args.config_dir)

    if args.output is not None:
        cfg.work_dir = args.output
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs',
                                os.path.splitext(os.path.basename(args.config))[0])
    
    if args.train:  # train mode
        # wandb를 사용하기 위한 hook 설정
        cfg.visualizer.vis_backends=[
            dict(
                type='WandbVisBackend',
                init_kwargs=dict(
                    project='Detr',
                    name=args.wandb_name),
                )
            ]
        
        # enable automatic-mixed-precision training
        if args.amp:
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'
            
        cfg.data_root = args.root
        cfg.train_dataloader.dataset.ann_file = args.annotation  # train json 정보
        cfg.val_dataloader.dataset.ann_file = args.valid_annotation  # validation json 정보
        cfg.load_from = args.load_from
        
    else:  # test(inference) mode
        cfg.load_from = os.path.join(cfg.work_dir, f"{args.checkpoint}.pth")
            
        cfg.data_root = args.root
        cfg.test_dataloader.dataset.ann_file = args.annotation  # test.json 정보
        cfg.test_evaluator.outfile_prefix = args.output
        
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    
    if args.train:
        runner.train()  # start training
    else:
        runner.test()  # start testing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # amp 사용여부 (default=False)
    parser.add_argument(
        '--amp',
        type=int,
        default=0,
        help='enable automatic-mixed-precision training')
    # 사전학습 가중치 가져오기
    parser.add_argument(
        '--load_from',
        type=str,
        default=None,
        help='load pre-trained model weight path, endswith:.pth')
    # 데이터셋 위치
    parser.add_argument(
        "--root", 
        type=str, 
        default="../../dataset/",
        help="dataset's location (default: ../../dataset/)",)
    # Annotation 파일 (학습 파일) 정보
    parser.add_argument(
        "--annotation", 
        type=str, 
        default="k-fold/train_fold_7.json",
        help="annotation file name (default: train.json)")
    parser.add_argument(
        "--valid_annotation", 
        type=str, 
        default="k-fold/val_fold_7.json",
        help="annotation file name (default: train.json)")
    # output 위치
    parser.add_argument(
        "--output", 
        type=str, 
        default="./work_dirs/default",
        help="output's location (default: ./work_dirs/default)")
    # train/test mode
    parser.add_argument(
        "--train", 
        type=int, 
        default=1,
        help="set inference/train mode, 0: inference / 1: train")
    # valid/submission mode
    parser.add_argument(
        "--valid", 
        type=int, 
        default=1,
        help="set submission/valid mode, 0: submission / 1: valid")
    # Config file
    parser.add_argument(
        "--config_dir", 
        type=str, 
        default=None,
        help="config file's location")
    # wandb name
    parser.add_argument(
        "--wandb_name", 
        type=str, 
        default=None,
        help="name of wandb test name")
    #################### TEST ####################
    # model pth name
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="latest",
        help="name of checkpoint want to inference")
    # Confidence Threshold
    parser.add_argument(
        "--conf_threshold", 
        type=float, 
        default=0.3,
        help="Confidence threshold used in confusion matrix")
    # IOU Threshold
    parser.add_argument(
        "--iou_threshold", 
        type=float, 
        default=0.5,
        help="IoU threshold used in confusion matrix and mAP")

    args = parser.parse_args()

    if args.config_dir == None:
        raise Exception("Import config file's location")
    if args.train and args.wandb_name == None:
        raise Exception("Import wandb test name")
    if args.load_from == None:
        raise Exception("Import pre-trained model weight path. endswith : .pth")
    if args.output == "./work_dirs/default":
        print(
            "Warning: Your output directory is set to (./work_dirs/default), you should change your output directory."
        )
    if args.checkpoint == "latest":
        print(
            "Warning: Your model name is set to (latest). If not intended, change your model name."
        )

    print(args)  # 내가 설정한 augrments
    
    main(args)