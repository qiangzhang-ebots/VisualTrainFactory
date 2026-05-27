

import multiprocessing
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


from ultralytics import YOLO


def _resolve_workspace(workspace):
    return Path(workspace).expanduser().resolve()


def _normalize_img_size(img_size):
    if isinstance(img_size, (tuple, list)):
        if len(img_size) != 2:
            raise ValueError('img_size must be an int or a 2-item sequence')
        return int(img_size[0]), int(img_size[1])
    size = int(img_size)
    return size, size


def _get_log_name():
    log_name = os.environ.get('TRAIN_LOGNAME')
    if log_name:
        return log_name

    log_name = 'train_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    os.environ['TRAIN_LOGNAME'] = log_name
    return log_name


def _get_gpu_ids(gpu_spec):
    return [item.strip() for item in str(gpu_spec).split(',') if item.strip()]


gpu = '0'
logName = _get_log_name()
is_torchrun_worker = os.environ.get('HRNET_DIST_LAUNCHED') == '1' and os.environ.get('LOCAL_RANK') is not None


def trainYolo(workspace, epochs, batch_size, img_size, weights=None):
    import yaml

    workspace_path = _resolve_workspace(workspace)
    script_dir = Path(__file__).resolve().parent
    yaml_path = workspace_path / 'FPC-pose.yaml'
    if not yaml_path.exists():
        yaml_path = script_dir / 'FPC-pose.yaml'
    if not yaml_path.exists():
        raise FileNotFoundError(f'YAML not found: {yaml_path}')

    with open(yaml_path, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}

    dataset_path = workspace_path / 'datasets'
    if data.get('path') != str(dataset_path):
        data['path'] = str(dataset_path)
        with open(yaml_path, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)

    model = YOLO(weights or 'yolo26n-pose.pt')
    model.train(
        data=str(yaml_path),
        epochs=int(epochs),
        imgsz=int(img_size),
        batch=int(batch_size),
        device=gpu,
        flipud=0.0,
        workers=8,
        name=logName,
        project=str(workspace_path / 'runs' / 'pose'),
        cache='ram',
        close_mosaic=20,
    )


def trainHRNet(workspace, epochs, batch_size, img_size):
    workspace_path = _resolve_workspace(workspace)
    script_dir = Path(__file__).resolve().parent
    dataset_root = workspace_path / 'datasets'
    train_ann = dataset_root / 'train_keypoints.json'
    val_ann = dataset_root / 'val_keypoints.json'
    work_dir = workspace_path / 'runs' / 'HRNet' / logName
    work_dir.mkdir(parents=True, exist_ok=True)

    if not train_ann.exists():
        raise FileNotFoundError(f'Train annotation not found: {train_ann}')
    if not val_ann.exists():
        raise FileNotFoundError(f'Val annotation not found: {val_ann}')

    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

    gpu_ids = _get_gpu_ids(gpu)
    use_multi_gpu = len(gpu_ids) > 1
    dist_launched = os.environ.get('HRNET_DIST_LAUNCHED') == '1'

    if use_multi_gpu and not dist_launched and os.environ.get('LOCAL_RANK') is None:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu
        env['HRNET_DIST_LAUNCHED'] = '1'
        env.setdefault('NCCL_DEBUG', 'WARN')
        subprocess.run(
            [
                sys.executable,
                '-m',
                'torch.distributed.run',
                '--nproc_per_node',
                str(len(gpu_ids)),
                '--master_port',
                '29501',
                __file__,
            ],
            check=True,
            env=env,
        )
        return

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', gpu)

    input_size = _normalize_img_size(img_size)
    heatmap_size = (max(1, input_size[0] // 4), max(1, input_size[1] // 4))
    bbox_shift_factor = 0.1
    bbox_scale_range = (0.8, 1.2)
    pretrained_ckpt = 'https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w48-8ef0771d.pth'

    keypoint_names = ('p1', 'p2', 'p3', 'p4')
    skeleton = ((0, 1), (1, 2), (2, 3), (3, 0))
    keypoint_colors = ([255, 80, 80], [80, 200, 255], [255, 200, 80], [80, 220, 120])
    link_colors = ([255, 255, 0], [255, 128, 0], [0, 200, 255], [120, 255, 120])

    swap_map = {
        'p1': 'p2',
        'p2': 'p1',
        'p3': 'p4',
        'p4': 'p3',
    }
    keypoint_info = {
        idx: dict(
            name=name,
            id=idx,
            color=list(keypoint_colors[idx]),
            type='upper',
            swap=swap_map[name])
        for idx, name in enumerate(keypoint_names)
    }
    skeleton_info = {
        idx: dict(
            link=(keypoint_names[start], keypoint_names[end]),
            id=idx,
            color=list(link_colors[idx]))
        for idx, (start, end) in enumerate(skeleton)
    }
    dataset_metainfo = dict(
        dataset_name='fpc_keypoints',
        keypoint_info=keypoint_info,
        skeleton_info=skeleton_info,
        joint_weights=[1.0] * len(keypoint_names),
        sigmas=[0.05] * len(keypoint_names),
    )

    codec = dict(
        type='UDPHeatmap',
        input_size=input_size,
        heatmap_size=heatmap_size,
        sigma=3,
    )
    train_pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(
            type='RandomBBoxTransform',
            shift_factor=bbox_shift_factor,
            scale_factor=list(bbox_scale_range),
            rotate_factor=0),
        dict(
            type='TopdownAffine',
            input_size=input_size,
            use_udp=True),
        dict(type='GenerateTarget', encoder=codec),
        dict(type='PackPoseInputs'),
    ]
    test_pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(
            type='TopdownAffine',
            input_size=input_size,
            use_udp=True),
        dict(type='PackPoseInputs'),
    ]

    epoch = int(epochs)
    batch = int(batch_size)
    val_batch = 1
    train_workers = 4
    val_workers = 2
    warmup_epochs = min(10, max(1, epoch // 10))
    lr = 5e-4
    logger_interval = 5
    checkpoint_interval = 1
    max_keep_ckpts = 10
    seed = 0

    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmpose.utils import register_all_modules

    cfg = Config(
        dict(
            default_scope='mmpose',
            work_dir=str(work_dir),
            launcher='pytorch' if use_multi_gpu else 'none',
            load_from=None,
            resume=False,
            log_level='INFO',
            randomness=dict(seed=seed, deterministic=False),
            env_cfg=dict(
                cudnn_benchmark=False,
                mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
                dist_cfg=dict(backend='nccl' if use_multi_gpu else 'gloo')),
            default_hooks=dict(
                timer=dict(type='IterTimerHook'),
                logger=dict(type='LoggerHook', interval=logger_interval),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(
                    type='CheckpointHook',
                    interval=checkpoint_interval,
                    max_keep_ckpts=max_keep_ckpts),
                sampler_seed=dict(type='DistSamplerSeedHook')),
            codec=codec,
            model=dict(
                type='TopdownPoseEstimator',
                data_preprocessor=dict(
                    type='PoseDataPreprocessor',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    bgr_to_rgb=True),
                backbone=dict(
                    type='HRNet',
                    in_channels=3,
                    extra=dict(
                        stage1=dict(
                            num_modules=1,
                            num_branches=1,
                            block='BOTTLENECK',
                            num_blocks=(4, ),
                            num_channels=(64, )),
                        stage2=dict(
                            num_modules=1,
                            num_branches=2,
                            block='BASIC',
                            num_blocks=(4, 4),
                            num_channels=(48, 96)),
                        stage3=dict(
                            num_modules=4,
                            num_branches=3,
                            block='BASIC',
                            num_blocks=(4, 4, 4),
                            num_channels=(48, 96, 192)),
                        stage4=dict(
                            num_modules=3,
                            num_branches=4,
                            block='BASIC',
                            num_blocks=(4, 4, 4, 4),
                            num_channels=(48, 96, 192, 384),
                            multiscale_output=True)),
                    init_cfg=dict(
                        type='Pretrained',
                        checkpoint=pretrained_ckpt)),
                neck=dict(
                    type='FeatureMapProcessor',
                    concat=True),
                head=dict(
                    type='HeatmapHead',
                    in_channels=48 + 96 + 192 + 384,
                    out_channels=len(keypoint_names),
                    deconv_out_channels=None,
                    loss=dict(type='KeypointMSELoss', use_target_weight=True),
                    decoder=codec),
                test_cfg=dict(
                    flip_test=False,
                    flip_mode='heatmap',
                    shift_heatmap=False)),
            train_dataloader=dict(
                batch_size=batch,
                num_workers=train_workers,
                persistent_workers=True,
                sampler=dict(type='DefaultSampler', shuffle=True),
                dataset=dict(
                    type='CocoDataset',
                    data_root='',
                    data_mode='topdown',
                    ann_file=str(train_ann),
                    data_prefix=dict(img=str(dataset_root / 'images' / 'train')),
                    metainfo=dataset_metainfo,
                    pipeline=train_pipeline)),
            val_dataloader=dict(
                batch_size=val_batch,
                num_workers=val_workers,
                persistent_workers=False,
                drop_last=False,
                sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
                dataset=dict(
                    type='CocoDataset',
                    data_root='',
                    data_mode='topdown',
                    ann_file=str(val_ann),
                    data_prefix=dict(img=str(dataset_root / 'images' / 'val')),
                    metainfo=dataset_metainfo,
                    test_mode=True,
                    pipeline=test_pipeline)),
            train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=epoch),
            val_cfg=None,
            test_cfg=None,
            param_scheduler=[
                dict(type='LinearLR', begin=0, end=warmup_epochs, start_factor=0.2, by_epoch=False),
                dict(type='CosineAnnealingLR', begin=0, end=epoch, T_max=epoch, eta_min=1e-5, by_epoch=True),
            ],
            optim_wrapper=dict(
                type='AmpOptimWrapper',
                optimizer=dict(type='Adam', lr=lr),
                loss_scale='dynamic'),
        ))

    runner_cfg = cfg.copy()
    runner_cfg.val_dataloader = None
    runner_cfg.val_cfg = None
    runner_cfg.val_evaluator = None
    runner_cfg.test_dataloader = None
    runner_cfg.test_cfg = None
    runner_cfg.test_evaluator = None

    register_all_modules(init_default_scope=True)
    runner = Runner.from_cfg(runner_cfg)
    runner.train()

    checkpoints = sorted(work_dir.glob('*.pth'))
    if checkpoints:
        print(f'HRNet training finished, latest checkpoint: {checkpoints[-1]}')
    else:
        print(f'HRNet training finished, but no checkpoint was found in: {work_dir}')


if __name__ == '__main__':
    multiprocessing.freeze_support()
