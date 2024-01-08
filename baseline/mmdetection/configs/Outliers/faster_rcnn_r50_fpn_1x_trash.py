_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu = 4
)

model = dict(
    roi_head = dict(
        bbox_head = dict(
            num_classes = 10
        )
    )
)

optimizer_config = dict(
    _delete_=True,
    grad_clip = dict(max_norm=35, norm_type=2)
)

checkpoint_config = dict(max_keep_ckpts=3, interval=1)

seed = 2022

gpu_ids = [0]