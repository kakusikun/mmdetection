_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='CenterNet',
    pretrained='',
    backbone=dict(
        type='MobileNetV2',
        out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        # norm_cfg=dict(type='BN', requires_grad=False),
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
        feat_channels={'hm': 1, 'wh': 2, 'offset': 2, 'orie': 6},
        stacked_convs=4,
        strides=(4, 8, 16, 32),
        dcn_on_last_conv=False,
        conv_bias='auto',
        loss_hm=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='SmoothL1Loss', loss_weight=1.0),
        loss_offset=dict(type='SmoothL1Loss', loss_weight=1.0),
        loss_orie=dict(type='SmoothL1Loss', loss_weight=1.0),)
)
    # bbox_head=dict(
    #     type='FCOSHead',
    #     num_classes=80,
    #     in_channels=256,
    #     stacked_convs=4,
    #     feat_channels=256,
    #     strides=[8, 16, 32, 64, 128],
    #     norm_cfg=None,
    #     loss_cls=dict(
    #         type='FocalLoss',
    #         use_sigmoid=True,
    #         gamma=2.0,
    #         alpha=0.25,
    #         loss_weight=1.0),
    #     loss_bbox=dict(type='IoULoss', loss_weight=1.0),
    #     loss_centerness=dict(
    #         type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100)
img_norm_cfg = dict(
    mean=[0.485*255, 0.456*255, 0.406*255], std=[0.229*255, 0.224*255, 0.225*255], to_rgb=True)
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
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineRestart',
    periods=[10,20,40],
    restart_weights=[1,1,1],
    min_lr=1e-7,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1.0 / 3,
    warmup_by_epoch=True)
total_epochs = 80
work_dir = './tensorboard/centerhand'
workflow = [('train', 1), ('val', 1)]