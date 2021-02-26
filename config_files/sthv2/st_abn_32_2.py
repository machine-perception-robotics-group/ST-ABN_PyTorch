model = dict(
    type='TSN3D',
    backbone=dict(
        type='ResNet_SlowFast',
        pretrained='modelzoo://resnet50',
        depth=50,
        #nsegments=8,
        #out_indices=(2, 3),
        #tsm=True,
        #bn_eval=False,
        #partial_bn=False),
        num_stages=4,
        #out_indices=[2, 3],
        out_indices=[3],
        frozen_stages=-1,
        inflate_freq=(0, 0, 1, 1),
        inflate_style='3x1x1',
        conv1_kernel_t=1,
        conv1_stride_t=1,
        pool1_kernel_t=1,
        pool1_stride_t=1,
        with_cp=True,
        bn_eval=False,
        partial_bn=False,
        style='pytorch'),
    necks=dict(
        type='Att_head',
        in_channels=[1024, 2048],
        out_channels=1024,
        spatial_modulation_config=dict(
            inplanes=[1024, 2048],
            planes=2048,
        ),
        temporal_modulation_config=dict(
            scales=(8, 8),
            param=dict(
                inplanes=-1,
                planes=-1,
                downsample_scale=-1,
            )),
        upsampling_config=dict(
            scale=(1, 1, 1),
        ),
        downsampling_config=dict(
            scales=(1, 1, 1),
            param=dict(
                inplanes=-1,
                planes=-1,
                downsample_scale=-1,
            )),
        level_fusion_config=dict(
            in_channels=[1024, 1024],
            mid_channels=[1024, 1024],
            out_channels=2048,
            ds_scales=[(1, 1, 1), (1, 1, 1)],
        ),
        att_head_config=dict(
            inplanes=-1,
            planes=174,
            loss_weight=0.5
        ),
    ),
    spatial_temporal_module=dict(
        type='SimpleSpatialTemporalModule',
        spatial_type='avg',
        temporal_size=32,
        spatial_size=7),
    segmental_consensus=dict(
        type='SimpleConsensus',
        consensus_type='avg'),

    cls_head=dict(
        type='ClsHead',
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.5,
        in_channels=2048,
        num_classes=174))
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'RawFramesDataset'
data_root = '/masa/TPN/data/smth_smth_v2/frames'
data_root_val = '/masa/TPN/data/smth_smth_v2/frames'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file='data/smth_smth_v2/anno/train_videofolder.txt',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        num_segments=1,
        new_length=32,
        new_step=2,
        random_shift=True,
        modality='RGB',
        image_tmpl='{:06d}.jpg',
        img_scale=256,
        input_size=224,
        flip_ratio=0.5,
        resize_keep_ratio=True,
        resize_crop=True,
        color_jitter=True,
        color_space_aug=True,
        oversample=None,
        max_distort=1,
        test_mode=False),
    val=dict(
        type=dataset_type,
        ann_file='data/smth_smth_v2/anno/val_videofolder.txt',
        img_prefix=data_root_val,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        num_segments=1,
        new_length=32,
        new_step=2,
        random_shift=False,
        modality='RGB',
        image_tmpl='{:06d}.jpg',
        img_scale=256,
        input_size=224,
        flip_ratio=0,
        resize_keep_ratio=True,
        oversample=None,
        test_mode=False),
    test=dict(
        type=dataset_type,
        ann_file='data/smth_smth_v2/anno/val_videofolder.txt',
        img_prefix=data_root_val,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        num_segments=1,
        new_length=32,
        new_step=2,
        random_shift=False,
        modality='RGB',
        image_tmpl='{:06d}.jpg',
        img_scale=256,
        input_size=224,
        flip_ratio=0,
        resize_keep_ratio=True,
        oversample="None",
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[75, 125])
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
# yapf:disable
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 150
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
