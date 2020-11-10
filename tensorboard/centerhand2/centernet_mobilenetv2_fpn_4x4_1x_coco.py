dataset_type = 'CocoHandDataset'
data_root = 'data/hand/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.120000000000005, 57.375],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.120000000000005, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_majors', 'gt_minors'])
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
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.120000000000005, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=8,
    train=dict(
        type='CocoHandDataset',
        ann_file='data/hand/annotations/instances_train2017.json',
        img_prefix='data/hand/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.120000000000005, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_majors', 'gt_minors'
                ])
        ]),
    val=dict(
        type='CocoHandDataset',
        ann_file='data/hand/annotations/instances_val2017.json',
        img_prefix='data/hand/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.120000000000005, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoHandDataset',
        ann_file='data/hand/annotations/instances_val2017.json',
        img_prefix='data/hand/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.120000000000005, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
train_cfg = None
test_cfg = None
model = dict(
    type='CenterNet',
    pretrained='',
    backbone=dict(type='MobileNetV2', out_indices=(0, 1, 2, 3)),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        num_outs=4,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CenterHandHead',
        in_channels=256,
        feat_channels=24,
        cls_channels=dict(hm=1, wh=2, offset=2, orie=6),
        stacked_convs=4,
        strides=(4, 8, 16, 32),
        dcn_on_last_conv=False,
        conv_bias='auto',
        loss_hm=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='SmoothL1Loss', loss_weight=1.0),
        loss_offset=dict(type='SmoothL1Loss', loss_weight=1.0),
        loss_orie=dict(type='SmoothL1Loss', loss_weight=1.0)))
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step', by_epoch=True, gamma=0.1, step=[90, 120])
total_epochs = 140
work_dir = './tensorboard/centerhand2'
gpu_ids = range(0, 1)
