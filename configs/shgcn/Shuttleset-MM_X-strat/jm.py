
dataset_type = 'PoseDataset'
ann_file = 'data/shuttleset_strat/train_100.pkl'  
val_ann_file = 'data/shuttleset_strat/val_100.pkl'  
test_ann_file = 'data/shuttleset_strat/test_100.pkl'  


train_pipeline = [
    dict(type='PoseDecode'),  
    dict(type='PreNormalize2D'), 
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),  
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']), 
    dict(type='GenBallFeat', use_vel=True, use_acc=True, use_angle=True, keep_conf=True),  
    dict(type='RandomTemporalShift', shift_prob=0.3, max_shift_ratio=0.3, pad_mode='edge'),
    dict(type='UniformSample_order', clip_len=100, num_clips=1, p_interval=(1, 1)), 
    dict(type='FormatGCNInput', num_person=2), 
    dict(type='Collect', keys=['keypoint', 'ball_trajectory', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint', 'ball_trajectory'])
]


val_pipeline = [
    dict(type='PoseDecode'),
    dict(type='PreNormalize2D'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(type='GenBallFeat', use_vel=True, use_acc=True, use_angle=True, keep_conf=True), 
    dict(type='UniformSample_order', clip_len=100, num_clips=1, p_interval=(1, 1)), 
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'ball_trajectory', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint', 'ball_trajectory'])
]

test_pipeline = [
    dict(type='PoseDecode'),
    dict(type='PreNormalize2D'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(type='GenBallFeat', use_vel=True, use_acc=True, use_angle=True, keep_conf=True), 
    dict(type='UniformSample_order', clip_len=100, num_clips=1, p_interval=(1, 1)), 
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'ball_trajectory', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint', 'ball_trajectory'])
]


data = dict(
    videos_per_gpu=64,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='PoseDataset',
            ann_file=ann_file,
            data_prefix='',
            pipeline=train_pipeline,
            split=None
        )
    ),
    val=dict(
        type='PoseDataset',
        ann_file=val_ann_file,
        data_prefix='',
        pipeline=val_pipeline,
        split=None
    ),
    test=dict(
        type='PoseDataset',
        ann_file=test_ann_file,
        data_prefix='',
        pipeline=test_pipeline,
        split=None
    )
)


optimizer = dict(
    type='SGD',
    lr=0.05,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True)
optimizer_config = dict(grad_clip=None)


lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs =100


work_dir = './work_dirs/shgcn/shuttleset_strat/jm'       
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])


evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])


model = dict(
    type='RecognizerGCN_CSCL',  
    backbone=dict(
        type='SHGCN',
        gcn_type='dgphgcn1',
        gcn_ratio=0.125,
        gcn_node_attention=True,
        gcn_edge_attention=True,
        gcn_decompose=True,
        gcn_subset_wise=True,
        gcn_ctr='T',
        gcn_ada='T',
        tcn_type='dgmsmlp',
        tcn_add_tcn=True,
        tcn_merge_after=True,
        num_person=2,
        graph_cfg=dict(
            layout='coco',
            mode='random',
            num_filter=3,
            init_off=0.04,
            init_std=0.02),
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']),
    cls_head=dict(
        type='GCNHead',
        num_classes=18,      
        in_channels=256,
        pool_mode='max',
        loss_cls=dict(type='CrossEntropyLoss', label_smoothing=0.1)),
    loss_weights=dict(skeleton=1.0, csc=0.4)    
)
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]