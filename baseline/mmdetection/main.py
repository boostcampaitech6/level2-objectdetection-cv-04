# Import module
import os
import os.path as osp
import argparse
# from mmdet.apis import single_gpu_test
# from mmcv.runner import load_checkpoint
# from mmcv.parallel import MMDataParallel
# from pycocotools.coco import COCO
# import pandas as pd
# from metrics import meanAveragePrecision, confusion_matrix

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def main(args):
    # config file 들고오기
    cfg = Config.fromfile(args.config_dir)

    cfg.work_dir = args.output
    
    if args.train:  # train mode
        # wandb를 사용하기 위한 hook 설정
        cfg.visualizer.vis_backends=[
            dict(
                type='WandbVisBackend', 
                init_kwargs=dict(
                    project='level2-object-detection-cv-04',
                    name=args.wandb_name),
                )
            ]
        
        cfg.data_root = args.root
        cfg.train_dataloader.dataset.data_prefix = dict(img="")
        cfg.train_dataloader.dataset.ann_file = args.annotation  # train json 정보
        cfg.val_dataloader.dataset.data_prefix = dict(img="")
        cfg.val_dataloader.dataset.ann_file = args.valid_annotation  # validation json 정보

        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        # start training
        runner.train()

    # else:  # test(inference) mode
    #     cfg.data.test.img_prefix = args.root
    #     cfg.data.test.ann_file = args.root + args.annotation  # test.json 정보
    #     cfg.data.test.test_mode = True

    #     # build dataset & dataloader
    #     dataset = build_dataset(cfg.data.test)
    #     data_loader = build_dataloader(
    #         dataset,
    #         samples_per_gpu=1,
    #         workers_per_gpu=cfg.data.workers_per_gpu,
    #         dist=False,
    #         shuffle=False,
    #     )

    #     # checkpoint path
    #     checkpoint_path = os.path.join(cfg.work_dir, f"{args.checkpoint}.pth")
    #     model = build_detector(
    #         cfg.model, test_cfg=cfg.get("test_cfg")
    #     )  # build detector
    #     checkpoint = load_checkpoint(
    #         model, checkpoint_path, map_location="cpu"
    #     )  # ckpt load
    #     model.CLASSES = dataset.CLASSES
    #     model = MMDataParallel(model.cuda(), device_ids=[0])
    #     output = single_gpu_test(model, data_loader, show_score_thr=0.05)  # output 계산

    #     prediction_strings = []
    #     file_names = []
    #     coco = COCO(cfg.data.test.ann_file)

    #     class_num = 10
    #     for i, out in zip(coco.getImgIds(), output):
    #         prediction_string = ""
    #         image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
    #         for j in range(class_num):
    #             for o in out[j]:
    #                 """
    #                 기존 out: shape: (5,)
    #                     x_min, y_min, x_max, y_max, confidence_score

    #                 -out -> prediction_string 재구성-
    #                     class_id, confidence_score, x_min, y_min, x_max, y_max
    #                 """
    #                 prediction_string += (
    #                     str(j)
    #                     + " "
    #                     + str(o[4])
    #                     + " "
    #                     + str(o[0])
    #                     + " "
    #                     + str(o[1])
    #                     + " "
    #                     + str(o[2])
    #                     + " "
    #                     + str(o[3])
    #                     + " "
    #                 )

    #         prediction_strings.append(prediction_string)
    #         file_names.append(image_info["file_name"])

    #     submission = pd.DataFrame()
    #     submission["PredictionString"] = prediction_strings
    #     submission["image_id"] = file_names

    #     if args.valid:
    #         mAP = meanAveragePrecision(submission, args)
    #         conf_matrix = confusion_matrix(submission, args)
    #         print(f"mAP(mean Average Precision): {mAP}")
    #         print(conf_matrix)
    #     else:
    #         submission.to_csv(
    #             os.path.join(cfg.work_dir, f"submission_{args.checkpoint}.csv"),
    #             index=None,
    #         )
    #         submission.head()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 데이터셋 위치
    parser.add_argument(
        "--root",
        type=str,
        default="../../dataset/",
        help="dataset's location (default: ../../dataset/)",
    )
    # Annotation 파일 (학습 파일) 정보
    parser.add_argument(
        "--annotation",
        type=str,
        default="k-fold/train_fold_7.json",
        help="annotation file name (default: train.json)",
    )
    parser.add_argument(
        "--valid_annotation",
        type=str,
        default="k-fold/val_fold_7.json",
        help="annotation file name (default: train.json)",
    )
    # output 위치
    parser.add_argument(
        "--output",
        type=str,
        default="./work_dirs/default",
        help="output's location (default: ./work_dirs/default)",
    )

    # train/test mode
    parser.add_argument(
        "--train",
        type=int,
        default=1,
        help="set inference/train mode, 0: inference / 1: train",
    )
    # valid/submission mode
    parser.add_argument(
        "--valid",
        type=int,
        default=1,
        help="set submission/valid mode, 0: submission / 1: valid",
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
        "--wandb_name", type=str, default=None, help="name of wandb test name"
    )

    # model pth name
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="name of checkpoint want to inference",
    )

    # Confidence Threshold
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.3,
        help="Confidence threshold used in confusion matrix",
    )

    # IOU Threshold
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold used in confusion matrix and mAP",
    )

    args = parser.parse_args()

    if args.config_dir == None:
        raise Exception("Import config file's location")

    if args.train and args.wandb_name == None:
        raise Exception("Import wandb test name")

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
