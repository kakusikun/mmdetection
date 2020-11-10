_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
dataset_type = 'CocoHandDataset'
data_root = 'data/hand/'
img_norm_cfg = dict(
    mean=[0.485*255, 0.456*255, 0.406*255], std=[0.229*255, 0.224*255, 0.225*255], to_rgb=True)
train_cfg = None
test_cfg = None

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_majors', 'gt_minors']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=10,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

# model settings
model = dict(
    type='CenterNet',
    pretrained='',
    backbone=dict(
        type='MobileNetV2',
        out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        # norm_cfg=dict(requires_grad=False),
        # norm_eval=True,
        # style='caffe'
    ),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=4,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CenterHandHead',
        in_channels=256,
        feat_channels=24,
        cls_channels={'hm': 1, 'wh': 2, 'offset': 2, 'orie': 6},
        stacked_convs=4,
        strides=(4, 8, 16, 32),
        dcn_on_last_conv=False,
        conv_bias='auto',
        loss_hm=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='SmoothL1Loss', loss_weight=1.0),
        loss_offset=dict(type='SmoothL1Loss', loss_weight=1.0),
        loss_orie=dict(type='SmoothL1Loss', loss_weight=1.0),)
)

# optimizer
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    by_epoch=True,
    gamma=0.1,
    step=[90, 120])
total_epochs = 140
work_dir = './tensorboard/centerhand2'
workflow = [('train', 1), ('val', 1)]