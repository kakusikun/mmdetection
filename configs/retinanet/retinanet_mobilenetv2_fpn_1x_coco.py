_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/hand_coco_detection.py',
    '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
#learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1.0/3.0,
    min_lr_ratio=1e-5,
    warmup_by_epoch=True)
total_epochs = 100