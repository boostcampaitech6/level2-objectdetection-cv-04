_base_ = [
    "../trash_detection.py",
    "../trash_runtime.py"
]

# 우리 데이터셋에 맞는 mean, std 값
Mean = [110.07, 117.39, 123.65]
Std = [54.77, 53.35, 54.01]

pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth"  # noqa
# LOAD_FROM="https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"  # 36 epochs
LOAD_FROM="https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth"  # 12 epoch

num_levels = 5

model = dict(
    type="DINO",
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    num_feature_levels=num_levels,
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=Mean,
        std=Std,
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type="SwinTransformer",
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained)),
    neck=dict(
        type="ChannelMapper",
        in_channels=[192, 384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=num_levels),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=num_levels,
                                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                                dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=num_levels,
                                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),
    bbox_head=dict(
        type="DINOHead",
        num_classes=10,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None,
                        num_dn_queries=100)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="FocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args={{_base_.backend_args}}),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type="RandomChoiceResize",
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type="PackDetInputs")
]
train_dataloader = dict(
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=0.000083,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)})
)

# learning policy
max_epochs = 12
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    # 1~4 epoch
    dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True, milestones=list(range(1, 4)), gamma=0.82999),
    # 5 epoch
    dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True, milestones=[4], gamma=1.748915),
    # 6 epoch
    dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True, milestones=[5], gamma=0.120482),
    # 7 epoch
    dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True, milestones=[6], gamma=8.3),
    # 8~10 epoch
    dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True, milestones=list(range(7, 10)), gamma=0.82999),
    # 11 epoch
    dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True, milestones=[10], gamma=1.748915),
    # 12 epoch
    dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True, milestones=[11], gamma=0.120482)
]