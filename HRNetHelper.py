from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Use physical GPU 1 for this script.
# After this, the process will see it as CUDA device 0.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import cv2
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

json_path = r'/home/ebots/Desktop/zhq/XiaomiFPCDetection/labeldata'

REPO_ROOT = Path(__file__).resolve().parent
# DATA_ROOT = Path(json_path)
WORK_DIR = REPO_ROOT / 'work_dirs' / 'hrnet_w48_topdown'
# IMAGE_SIZE = 160
INPUT_SIZE = (512, 512)
HEATMAP_SIZE = (128, 128)
DEFAULT_DECODER = 'udp'
# TRAIN_SAMPLES = 8
# VAL_SAMPLES = 2
# DEFAULT_EPOCHS = 300

# Make top-down pose estimation more robust to imperfect detector boxes.
BBOX_SHIFT_FACTOR = 0.1
BBOX_SCALE_RANGE = (0.8, 1.2)

KEYPOINT_NAMES = ('top_left', 'top_right', 'bottom_right', 'bottom_left')
SKELETON = ((0, 1), (1, 2), (2, 3), (3, 0))
KEYPOINT_COLORS = ([255, 80, 80], [80, 200, 255], [255, 200, 80], [80, 220, 120])
LINK_COLORS = ([255, 255, 0], [255, 128, 0], [0, 200, 255], [120, 255, 120])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MMPose CPU demo: build data, train, and run inference.')
    parser.add_argument(
        '--task',
        choices=('all', 'prepare', 'train', 'infer'),
        default='infer',
        help='Which stage to run. "all" will prepare data, train, and infer.')
    parser.add_argument(
        '--epochs',
        type=int,
        default=300,
        help='Number of training epochs for the tiny demo dataset.')
    parser.add_argument(
        '--force-data',
        action='store_true',
        help='Recreate the synthetic demo dataset even if it already exists.')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.5,
        help='Visibility threshold used when saving predicted keypoints.')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help='Checkpoint path for inference. Defaults to the latest checkpoint in work_dirs.')
    parser.add_argument(
        '--decoder',
        choices=('udp', 'dark'),
        default=DEFAULT_DECODER,
        help='Use UDP or DARK-style decoding. UDP is the default.')
    parser.add_argument(
        '--save-heatmaps',
        action='store_true',
        help='Save predicted heatmaps alongside keypoint outputs during inference.')
    return parser.parse_args()


def build_dataset_metainfo() -> dict:
    swap_map = {
        'top_left': 'top_right',
        'top_right': 'top_left',
        'bottom_right': 'bottom_left',
        'bottom_left': 'bottom_right',
    }
    keypoint_info = {
        idx: dict(
            name=name,
            id=idx,
            color=list(KEYPOINT_COLORS[idx]),
            type='upper',
            swap=swap_map[name])
        for idx, name in enumerate(KEYPOINT_NAMES)
    }
    skeleton_info = {
        idx: dict(
            link=(KEYPOINT_NAMES[start], KEYPOINT_NAMES[end]),
            id=idx,
            color=list(LINK_COLORS[idx]))
        for idx, (start, end) in enumerate(SKELETON)
    }
    return dict(
        dataset_name='demo_rectangle_keypoints',
        keypoint_info=keypoint_info,
        skeleton_info=skeleton_info,
        joint_weights=[1.0] * len(KEYPOINT_NAMES),
        sigmas=[0.05] * len(KEYPOINT_NAMES),
    )


def build_demo_dataset(force_rebuild: bool = False) -> Dict[str, Path]:
    train_dir = json_path 
    val_dir = json_path 

    train_ann = './data/train_keypoints.json'
    val_ann = './data/val_keypoints.json'

    if not force_rebuild:
        return {
            'train_ann': train_ann,
            'val_ann': val_ann,
            'train_dir': train_dir,
            'val_dir': val_dir,
        }

    return {
        'train_ann': train_ann,
        'val_ann': val_ann,
        'train_dir': train_dir,
        'val_dir': val_dir,
    }


def build_codec(decoder: str) -> dict:
    if decoder == 'dark':
        return dict(
            type='MSRAHeatmap',
            input_size=INPUT_SIZE,
            heatmap_size=HEATMAP_SIZE,
            sigma=3,
            unbiased=True)

    return dict(
        type='UDPHeatmap',
        input_size=INPUT_SIZE,
        heatmap_size=HEATMAP_SIZE,
        sigma=3)


def build_test_cfg(decoder: str) -> dict:
    return dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=(decoder == 'dark'))


def build_config(max_epochs: int, decoder: str) -> Config:
    dataset_metainfo = build_dataset_metainfo()
    codec = build_codec(decoder)
    use_udp = decoder == 'udp'
    train_pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(
            type='RandomBBoxTransform',
            shift_factor=BBOX_SHIFT_FACTOR,
            scale_factor=list(BBOX_SCALE_RANGE),
            rotate_factor=0),
        dict(
            type='TopdownAffine',
            input_size=INPUT_SIZE,
            use_udp=use_udp),
        dict(type='GenerateTarget', encoder=codec),
        dict(type='PackPoseInputs'),
    ]
    test_pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(
            type='TopdownAffine',
            input_size=INPUT_SIZE,
            use_udp=use_udp),
        dict(type='PackPoseInputs'),
    ]

    cfg = Config(
        dict(
            default_scope='mmpose',
            work_dir=str(WORK_DIR),
            launcher='none',
            load_from=None,
            resume=False,
            log_level='INFO',
            randomness=dict(seed=0, deterministic=False),
            env_cfg=dict(
                cudnn_benchmark=False,
                mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
                dist_cfg=dict(backend='gloo')),
            default_hooks=dict(
                timer=dict(type='IterTimerHook'),
                logger=dict(type='LoggerHook', interval=5),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
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
                        checkpoint='https://download.openmmlab.com/mmpose/'
                        'pretrain_models/hrnet_w48-8ef0771d.pth')),
                neck=dict(
                    type='FeatureMapProcessor',
                    concat=True),
                head=dict(
                    type='HeatmapHead',
                    in_channels=48 + 96 + 192 + 384,
                    out_channels=len(KEYPOINT_NAMES),
                    deconv_out_channels=None,
                    loss=dict(type='KeypointMSELoss', use_target_weight=True),
                    decoder=codec),
                test_cfg=build_test_cfg(decoder)),
            train_dataloader=dict(
                batch_size=16,
                num_workers=4,
                persistent_workers=True,
                sampler=dict(type='DefaultSampler', shuffle=True),
                dataset=dict(
                    type='CocoDataset',
                    data_root='',
                    data_mode='topdown',
                    ann_file='./data/train_keypoints.json',
                    data_prefix=dict(img=json_path),
                    metainfo=dataset_metainfo,
                    pipeline=train_pipeline)),
            test_dataloader=dict(
                batch_size=1,
                num_workers=2,
                persistent_workers=False,
                drop_last=False,
                sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
                dataset=dict(
                    type='CocoDataset',
                    data_root='',
                    data_mode='topdown',
                    ann_file='./data/val_keypoints.json',
                    data_prefix=dict(img=json_path),
                    metainfo=dataset_metainfo,
                    test_mode=True,
                    pipeline=test_pipeline)),
            train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=max_epochs),
            val_cfg=None,
            test_cfg=None,
            param_scheduler=[
                dict(type='LinearLR', begin=0, end=10, start_factor=0.2, by_epoch=False),
                dict(type='CosineAnnealingLR', begin=0, end=max_epochs, T_max=max_epochs, eta_min=1e-5, by_epoch=True),
            ],
            optim_wrapper=dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=5e-4)),
        ))
    return cfg


def train_pose_model(cfg: Config) -> Path:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    register_all_modules(init_default_scope=True)
    runner_cfg = cfg.copy()
    runner_cfg.val_dataloader = None
    runner_cfg.val_cfg = None
    runner_cfg.val_evaluator = None
    runner_cfg.test_dataloader = None
    runner_cfg.test_cfg = None
    runner_cfg.test_evaluator = None
    runner = Runner.from_cfg(runner_cfg)
    runner.train()
    checkpoint = find_latest_checkpoint(WORK_DIR)
    if checkpoint is None:
        raise FileNotFoundError(f'No checkpoint was produced in {WORK_DIR}')
    return checkpoint


def find_latest_checkpoint(work_dir: Path) -> Path | None:
    last_checkpoint = work_dir / 'last_checkpoint'
    if last_checkpoint.exists():
        checkpoint_path = last_checkpoint.read_text(encoding='utf-8').strip()
        if checkpoint_path:
            return Path(checkpoint_path)

    checkpoints = sorted(work_dir.glob('*.pth'))
    if checkpoints:
        return checkpoints[-1]
    return None


def to_numpy(data):
    if hasattr(data, 'detach'):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def enable_output_heatmaps(model) -> None:
    if getattr(model, 'test_cfg', None) is None:
        model.test_cfg = {}
    model.test_cfg['output_heatmaps'] = True

    model_cfg = getattr(model, 'cfg', None)
    if model_cfg is not None and 'model' in model_cfg:
        test_cfg = model_cfg.model.get('test_cfg', {})
        test_cfg['output_heatmaps'] = True
        model_cfg.model['test_cfg'] = test_cfg


def extract_pred_heatmaps(results) -> np.ndarray | None:
    batch_heatmaps = []
    for result in results:
        pred_fields = getattr(result, 'pred_fields', None)
        heatmaps = getattr(pred_fields, 'heatmaps', None)
        if heatmaps is None:
            return None
        batch_heatmaps.append(to_numpy(heatmaps))

    if not batch_heatmaps:
        return None

    return np.stack(batch_heatmaps, axis=0)


def jitter_bboxes_xywh(
    bboxes: List[List[float]],
    image_size: Tuple[int, int] | None = None,
    shift_factor: float = BBOX_SHIFT_FACTOR,
    scale_range: Tuple[float, float] = BBOX_SCALE_RANGE,
) -> List[List[float]]:
    if shift_factor <= 0 and np.isclose(scale_range[0], 1.0) and np.isclose(scale_range[1], 1.0):
        return [list(map(float, bbox)) for bbox in bboxes]

    image_width = image_height = None
    if image_size is not None:
        image_width, image_height = image_size

    perturbed_bboxes: List[List[float]] = []
    for bbox in bboxes:
        x, y, w, h = map(float, bbox)
        if w <= 0 or h <= 0:
            perturbed_bboxes.append([x, y, w, h])
            continue

        center_x = x + w / 2.0
        center_y = y + h / 2.0
        scale = float(np.random.uniform(scale_range[0], scale_range[1]))
        shift_x = float(np.random.uniform(-shift_factor, shift_factor) * w)
        shift_y = float(np.random.uniform(-shift_factor, shift_factor) * h)

        new_w = max(1.0, w * scale)
        new_h = max(1.0, h * scale)
        new_x = center_x + shift_x - new_w / 2.0
        new_y = center_y + shift_y - new_h / 2.0

        if image_width is not None and image_height is not None:
            new_x = float(np.clip(new_x, 0.0, max(0.0, image_width - 1.0)))
            new_y = float(np.clip(new_y, 0.0, max(0.0, image_height - 1.0)))
            new_w = float(min(new_w, max(1.0, image_width - new_x)))
            new_h = float(min(new_h, max(1.0, image_height - new_y)))

        perturbed_bboxes.append([new_x, new_y, new_w, new_h])

    return perturbed_bboxes


def load_inference_sample(annotation_path: Path) -> Tuple[Path, List[List[float]]]:
    annotation_data = json.loads(annotation_path.read_text(encoding='utf-8'))
    images = {image['id']: image for image in annotation_data['images']}
    annotations_by_image: Dict[int, List[dict]] = {}

    for annotation in annotation_data['annotations']:
        annotations_by_image.setdefault(annotation['image_id'], []).append(annotation)

    for image_id in sorted(images):
        image_annotations = annotations_by_image.get(image_id, [])
        if not image_annotations:
            continue

        image_path = Path(json_path) / images[image_id]['file_name']
        bboxes = [annotation['bbox'] for annotation in image_annotations]
        return image_path, bboxes

    raise RuntimeError(f'No annotated validation samples found in {annotation_path}')


def group_annotations_by_image(annotation_data: dict) -> Dict[int, List[dict]]:
    annotations_by_image: Dict[int, List[dict]] = {}
    for annotation in annotation_data['annotations']:
        image_id = int(annotation['image_id'])
        annotations_by_image.setdefault(image_id, []).append(annotation)
    return annotations_by_image


def coco_keypoints_to_array(keypoints: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    keypoint_array = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)
    return keypoint_array[:, :2], keypoint_array[:, 2]


def summarize_error_values(values: List[float]) -> dict:
    if not values:
        return dict(
            count=0,
            mean=None,
            std=None,
            median=None,
            p995=None,
            min=None,
            max=None)

    array = np.asarray(values, dtype=np.float32)
    return dict(
        count=int(array.size),
        mean=round(float(array.mean()), 4),
        std=round(float(array.std()), 4),
        median=round(float(np.median(array)), 4),
        p995=round(float(np.percentile(array, 99.5)), 4),
        min=round(float(array.min()), 4),
        max=round(float(array.max()), 4),
    )


def build_labelme_polygon(category_id: str, points: np.ndarray) -> dict:
    return dict(
        label=category_id,
        points=[[float(point[0]), float(point[1])] for point in points],
        group_id=None,
        description='',
        shape_type='polygon',
        flags={},
        mask=None,
    )


def draw_error_histograms(errors_by_category: Dict[str, List[float]], output_path: Path) -> None:
    canvas_height = 720
    canvas_width = 1280
    panel_margin = 40
    panel_gap = 40
    panel_width = (canvas_width - panel_margin * 2 - panel_gap) // 2
    panel_height = 520
    plot_left_padding = 65
    plot_bottom_padding = 60
    plot_top_padding = 45
    plot_right_padding = 20
    title_y = 36

    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        'Validation Error Histograms by Category',
        (panel_margin, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    all_errors = [error for values in errors_by_category.values() for error in values]
    max_error = max(all_errors) if all_errors else 1.0
    if max_error <= 0:
        max_error = 1.0
    bins = np.linspace(0.0, max_error, 21)
    if np.allclose(bins[0], bins[-1]):
        bins = np.linspace(0.0, max_error + 1.0, 21)

    category_colors = {'0': (80, 160, 240), '1': (80, 200, 120)}
    mean_line_color = (60, 60, 220)
    median_line_color = (30, 150, 30)
    p995_line_color = (220, 120, 30)
    for panel_index, category_id in enumerate(('0', '1')):
        x0 = panel_margin + panel_index * (panel_width + panel_gap)
        y0 = 80
        x1 = x0 + panel_width
        y1 = y0 + panel_height

        cv2.rectangle(canvas, (x0, y0), (x1, y1), (220, 220, 220), 1)
        cv2.putText(
            canvas,
            f'Category {category_id}',
            (x0 + 10, y0 + title_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )

        values = np.asarray(errors_by_category.get(category_id, []), dtype=np.float32)
        plot_x0 = x0 + plot_left_padding
        plot_y0 = y0 + plot_top_padding
        plot_x1 = x1 - plot_right_padding
        plot_y1 = y1 - plot_bottom_padding

        cv2.line(canvas, (plot_x0, plot_y1), (plot_x1, plot_y1), (60, 60, 60), 1)
        cv2.line(canvas, (plot_x0, plot_y1), (plot_x0, plot_y0), (60, 60, 60), 1)

        if values.size == 0:
            cv2.putText(
                canvas,
                'No samples',
                (x0 + 110, y0 + 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (120, 120, 120),
                2,
                cv2.LINE_AA,
            )
            continue

        hist, edges = np.histogram(values, bins=bins)
        max_count = int(hist.max()) if hist.size else 1
        max_count = max(max_count, 1)
        bar_width = max((plot_x1 - plot_x0) // len(hist), 6)
        color = category_colors[category_id]

        for index, count in enumerate(hist):
            bar_left = plot_x0 + index * bar_width + 2
            bar_right = plot_x0 + (index + 1) * bar_width - 2
            bar_height = int(round((count / max_count) * (plot_y1 - plot_y0)))
            bar_top = plot_y1 - bar_height
            if bar_right <= bar_left:
                bar_right = bar_left + 1
            cv2.rectangle(canvas, (bar_left, bar_top), (bar_right, plot_y1), color, -1)

        stats = summarize_error_values(values.tolist())
        value_range = max(float(bins[-1]), 1e-6)
        mean_x = int(round(plot_x0 + (stats['mean'] / value_range) * (plot_x1 - plot_x0)))
        median_x = int(round(plot_x0 + (stats['median'] / value_range) * (plot_x1 - plot_x0)))
        p995_x = int(round(plot_x0 + (stats['p995'] / value_range) * (plot_x1 - plot_x0)))
        mean_x = max(plot_x0, min(plot_x1, mean_x))
        median_x = max(plot_x0, min(plot_x1, median_x))
        p995_x = max(plot_x0, min(plot_x1, p995_x))

        cv2.line(canvas, (mean_x, plot_y0), (mean_x, plot_y1), mean_line_color, 2)
        cv2.line(canvas, (median_x, plot_y0), (median_x, plot_y1), median_line_color, 2)
        cv2.line(canvas, (p995_x, plot_y0), (p995_x, plot_y1), p995_line_color, 2)

        cv2.putText(
            canvas,
            'mean',
            (max(plot_x0, mean_x - 18), plot_y0 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            mean_line_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            'median',
            (max(plot_x0, median_x - 24), plot_y0 + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            median_line_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            'p99.5',
            (max(plot_x0, p995_x - 22), plot_y0 + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            p995_line_color,
            1,
            cv2.LINE_AA,
        )

        for tick_index in range(5):
            ratio = tick_index / 4
            tick_y = int(round(plot_y1 - ratio * (plot_y1 - plot_y0)))
            tick_value = int(round(ratio * max_count))
            cv2.line(canvas, (plot_x0 - 5, tick_y), (plot_x0, tick_y), (60, 60, 60), 1)
            cv2.putText(
                canvas,
                str(tick_value),
                (x0 + 5, tick_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )

        for tick_index in range(5):
            ratio = tick_index / 4
            tick_x = int(round(plot_x0 + ratio * (plot_x1 - plot_x0)))
            tick_value = ratio * bins[-1]
            cv2.line(canvas, (tick_x, plot_y1), (tick_x, plot_y1 + 5), (60, 60, 60), 1)
            cv2.putText(
                canvas,
                f'{tick_value:.1f}',
                (tick_x - 12, plot_y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            (f"n={stats['count']} mean={stats['mean']:.2f} std={stats['std']:.2f} "
             f"median={stats['median']:.2f} p99.5={stats['p995']:.2f}"),
            (x0 + 10, y1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def run_inference(
    cfg: Config,
    checkpoint_path: Path,
    score_thr: float,
    save_heatmaps: bool = False,
) -> Path:
    register_all_modules(init_default_scope=True)
    image_path, bboxes = load_inference_sample(REPO_ROOT / 'data' / 'val_keypoints.json')
    image = cv2.imread(str(image_path))
    image_size = (int(image.shape[1]), int(image.shape[0])) if image is not None else None
    bboxes = jitter_bboxes_xywh(bboxes, image_size=image_size)
    model = init_model(cfg, str(checkpoint_path), device='cuda:0')
    enable_output_heatmaps(model)
    results = inference_topdown(
        model,
        str(image_path),
        bboxes=bboxes,
        bbox_format='xywh')
    if not results:
        raise RuntimeError('Inference returned no pose predictions.')

    heatmaps = extract_pred_heatmaps(results)

    keypoints = np.stack(
        [to_numpy(result.pred_instances.keypoints)[0] for result in results],
        axis=0)
    scores = np.stack(
        [to_numpy(result.pred_instances.keypoint_scores)[0] for result in results],
        axis=0)

    predictions = []
    for instance_index, (instance_keypoints,
                         instance_scores) in enumerate(zip(keypoints, scores)):
        pred_instances = results[instance_index].pred_instances
        if hasattr(pred_instances, 'scores'):
            instance_score = float(to_numpy(pred_instances.scores)[0])
        elif hasattr(pred_instances, 'bbox_scores'):
            instance_score = float(to_numpy(pred_instances.bbox_scores)[0])
        else:
            instance_score = float(np.mean(instance_scores))

        instance_prediction = {
            'instance_index': instance_index,
            'score': round(instance_score, 4),
            'keypoints': []
        }
        for name, point, score in zip(KEYPOINT_NAMES, instance_keypoints,
                                      instance_scores):
            instance_prediction['keypoints'].append({
                'name': name,
                'x': round(float(point[0]), 2),
                'y': round(float(point[1]), 2),
                'score': round(float(score), 4),
                'visible': bool(score >= score_thr),
            })
        predictions.append(instance_prediction)

    output_dir = WORK_DIR / 'predictions'
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = None
    if save_heatmaps and heatmaps is not None:
        heatmap_path = output_dir / f'{image_path.stem}_heatmaps.npz'
        np.savez_compressed(str(heatmap_path), heatmaps=heatmaps)

    output_json = output_dir / 'prediction_summary.json'
    output_json.write_text(
        json.dumps(
            {
                'image': image_path.name,
                'checkpoint': str(checkpoint_path),
                'num_objects': len(predictions),
                'heatmap_path': (str(heatmap_path) if heatmap_path is not None else None),
                'predictions': predictions,
            },
            indent=2),
        encoding='utf-8')

    output_image = output_dir / f'{image_path.stem}_keypoints.jpg'
    render_predictions(image_path, output_image, keypoints, scores, score_thr)
    return output_json


def run_inference_on_all_samples(
    cfg: Config,
    checkpoint_path: Path,
    score_thr: float,
    save_heatmaps: bool = False,
) -> Path:
    register_all_modules(init_default_scope=True)

    annotation_path = REPO_ROOT / 'data' / 'val_keypoints.json'
    annotation_data = json.loads(annotation_path.read_text(encoding='utf-8'))
    images = {int(image['id']): image for image in annotation_data['images']}
    annotations_by_image = group_annotations_by_image(annotation_data)

    val_dir = REPO_ROOT / 'Val'
    infer_dir = REPO_ROOT / 'Infere'
    heatmap_dir = infer_dir / 'heatmaps'
    visual_dir = infer_dir / 'visualizations'
    val_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)
    if save_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    model = init_model(cfg, str(checkpoint_path), device='cuda:0')
    enable_output_heatmaps(model)
    point_errors_by_category: Dict[str, List[float]] = {'0': [], '1': []}
    object_errors_by_category: Dict[str, List[float]] = {'0': [], '1': []}
    all_csv_rows = []
    inference_times_ms = []
    image_summaries = []

    for image_id in sorted(images):
        image_info = images[image_id]
        image_name = image_info['file_name']
        source_image_path = Path(json_path) / image_name
        if not source_image_path.exists():
            raise FileNotFoundError(f'Validation image not found: {source_image_path}')

        local_image_path = val_dir / image_name
        if source_image_path.resolve() != local_image_path.resolve():
            shutil.copy2(source_image_path, local_image_path)

        image_annotations = annotations_by_image.get(image_id, [])
        if not image_annotations:
            continue

        bboxes = jitter_bboxes_xywh(
            [annotation['bbox'] for annotation in image_annotations],
            image_size=(int(image_info['width']), int(image_info['height'])))
        inference_start_time = time.perf_counter()
        results = inference_topdown(
            model,
            str(local_image_path),
            bboxes=bboxes,
            bbox_format='xywh')
        inference_elapsed_ms = (time.perf_counter() - inference_start_time) * 1000.0
        inference_times_ms.append(inference_elapsed_ms)

        if not results:
            continue
        
        img = cv2.imread(local_image_path)
        heatmaps = extract_pred_heatmaps(results)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(img)
        # plt.ion()
        # plt.show()

        for hs in heatmaps:
            for h in hs:
                heatmap = h
                a = 1

        heatmap_path = None
        if save_heatmaps and heatmaps is not None:
            heatmap_path = heatmap_dir / f'{local_image_path.stem}_heatmaps.npz'
            np.savez_compressed(str(heatmap_path), heatmaps=heatmaps)

        keypoints = np.stack(
            [to_numpy(result.pred_instances.keypoints)[0] for result in results],
            axis=0)
        scores = np.stack(
            [to_numpy(result.pred_instances.keypoint_scores)[0] for result in results],
            axis=0)

        labelme_shapes = []
        prediction_details = []
        num_instances = min(len(image_annotations), keypoints.shape[0])
        for instance_index in range(num_instances):
            annotation = image_annotations[instance_index]
            category_id = str(annotation['category_id'])
            gt_points, gt_visibility = coco_keypoints_to_array(annotation['keypoints'])
            pred_points = keypoints[instance_index]
            pred_scores = scores[instance_index]
            raw_point_errors = np.linalg.norm(pred_points - gt_points, axis=1)
            valid_mask = (gt_visibility > 0) & (pred_scores >= score_thr)
            valid_point_errors = raw_point_errors[valid_mask]
            mean_error = (float(valid_point_errors.mean())
                          if valid_point_errors.size > 0 else None)

            point_errors_by_category.setdefault(category_id, []).extend(
                valid_point_errors.tolist())
            if mean_error is not None:
                object_errors_by_category.setdefault(category_id, []).append(mean_error)
            labelme_shapes.append(build_labelme_polygon(category_id, pred_points))

            prediction_details.append(
                dict(
                    annotation_id=int(annotation['id']),
                    category_id=category_id,
                    bbox=[float(value) for value in annotation['bbox']],
                    mean_error=(round(mean_error, 4) if mean_error is not None else None),
                    valid_points=int(valid_mask.sum()),
                    keypoints=[
                        dict(
                            name=name,
                            gt=[round(float(gt_point[0]), 2), round(float(gt_point[1]), 2)],
                            pred=[round(float(pred_point[0]), 2), round(float(pred_point[1]), 2)],
                            score=round(float(pred_score), 4),
                            visible=bool(gt_visible > 0),
                            found=bool(pred_score >= score_thr),
                            error=(round(float(error), 4)
                                   if pred_score >= score_thr and gt_visible > 0 else None),
                        )
                        for name, gt_point, pred_point, pred_score, gt_visible, error in zip(
                            KEYPOINT_NAMES,
                            gt_points,
                            pred_points,
                            pred_scores,
                            gt_visibility,
                            raw_point_errors,
                        )
                    ],
                ))

            for keypoint_name, gt_point, pred_point, pred_score, gt_visible, error in zip(
                    KEYPOINT_NAMES,
                    gt_points,
                    pred_points,
                    pred_scores,
                    gt_visibility,
                    raw_point_errors):
                all_csv_rows.append({
                    'image_name': image_name,
                    'image_id': image_id,
                    'annotation_id': int(annotation['id']),
                    'instance_index': instance_index,
                    'category_id': category_id,
                    'keypoint_name': keypoint_name,
                    'gt_x': round(float(gt_point[0]), 4),
                    'gt_y': round(float(gt_point[1]), 4),
                    'pred_x': round(float(pred_point[0]), 4),
                    'pred_y': round(float(pred_point[1]), 4),
                    'score': round(float(pred_score), 6),
                    'visible': int(gt_visible > 0),
                    'found': int(pred_score >= score_thr),
                    'point_error': (round(float(error), 6)
                                    if pred_score >= score_thr and gt_visible > 0 else None),
                    'object_mean_error': (round(mean_error, 6)
                                          if mean_error is not None else None),
                })

        labelme_output = dict(
            version='5.11.2.dev27+g31338d998.d20260304',
            flags={},
            shapes=labelme_shapes,
            imagePath=local_image_path.name,
            imageData=None,
            imageHeight=int(image_info['height']),
            imageWidth=int(image_info['width']),
        )
        output_json = infer_dir / f'{local_image_path.stem}.json'
        output_json.write_text(
            json.dumps(labelme_output, indent=2),
            encoding='utf-8')

        render_predictions(
            local_image_path,
            visual_dir / f'{local_image_path.stem}_keypoints.jpg',
            keypoints[:num_instances],
            scores[:num_instances],
            score_thr,
        )

        image_summaries.append(
            dict(
                image=image_name,
                image_id=image_id,
                num_annotations=len(image_annotations),
                num_predictions=int(num_instances),
                inference_time_ms=round(inference_elapsed_ms, 4),
                output_json=str(output_json),
                heatmap_path=(str(heatmap_path) if heatmap_path is not None else None),
                predictions=prediction_details,
            ))

    all_point_errors = [
        error
        for values in point_errors_by_category.values()
        for error in values
    ]
    all_object_errors = [
        error
        for values in object_errors_by_category.values()
        for error in values
    ]

    histogram_path = infer_dir / 'error_histogram_by_category.png'
    draw_error_histograms(point_errors_by_category, histogram_path)

    aggregated_csv_path = infer_dir / 'all_errors.csv'
    with aggregated_csv_path.open('w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                'image_name',
                'image_id',
                'annotation_id',
                'instance_index',
                'category_id',
                'keypoint_name',
                'gt_x',
                'gt_y',
                'pred_x',
                'pred_y',
                'score',
                'visible',
                'found',
                'point_error',
                'object_mean_error',
            ])
        writer.writeheader()
        writer.writerows(all_csv_rows)

    summary = dict(
        checkpoint=str(checkpoint_path),
        num_images=len(image_summaries),
        infer_dir=str(infer_dir),
        val_dir=str(val_dir),
        aggregated_csv_path=str(aggregated_csv_path),
        histogram_path=str(histogram_path),
        inference_time_ms=dict(
            count=len(inference_times_ms),
            mean=(round(float(np.mean(inference_times_ms)), 4)
                  if inference_times_ms else None),
            median=(round(float(np.median(inference_times_ms)), 4)
                    if inference_times_ms else None),
            min=(round(float(np.min(inference_times_ms)), 4)
                 if inference_times_ms else None),
            max=(round(float(np.max(inference_times_ms)), 4)
                 if inference_times_ms else None),
        ),
        point_error_stats=dict(
            overall=summarize_error_values(all_point_errors),
            by_category={
                category_id: summarize_error_values(values)
                for category_id, values in point_errors_by_category.items()
            },
        ),
        object_error_stats=dict(
            overall=summarize_error_values(all_object_errors),
            by_category={
                category_id: summarize_error_values(values)
                for category_id, values in object_errors_by_category.items()
            },
        ),
        images=image_summaries,
    )

    summary_path = infer_dir / 'summary.json'
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding='utf-8')
    return summary_path


def render_predictions(
    image_path: Path,
    output_path: Path,
    keypoints: np.ndarray,
    scores: np.ndarray,
    score_thr: float,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f'Unable to read image: {image_path}')

    for instance_index, (instance_keypoints,
                         instance_scores) in enumerate(zip(keypoints, scores)):
        color_shift = instance_index % len(KEYPOINT_COLORS)

        for link_index, (start, end) in enumerate(SKELETON):
            if instance_scores[start] < score_thr or instance_scores[end] < score_thr:
                continue
            start_point = tuple(
                int(round(value)) for value in instance_keypoints[start])
            end_point = tuple(
                int(round(value)) for value in instance_keypoints[end])
            cv2.line(image, start_point, end_point,
                     LINK_COLORS[(link_index + color_shift) % len(LINK_COLORS)],
                     thickness=2)

        for idx, (point, score) in enumerate(zip(instance_keypoints,
                                                 instance_scores)):
            if score < score_thr:
                continue
            center = tuple(int(round(value)) for value in point)
            cv2.circle(image, center, 4,
                       KEYPOINT_COLORS[(idx + color_shift) % len(KEYPOINT_COLORS)],
                       thickness=-1)
            cv2.putText(
                image,
                f'{instance_index}:{KEYPOINT_NAMES[idx]}:{score:.2f}',
                (center[0] + 4, max(14, center[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (30, 30, 30),
                1,
                cv2.LINE_AA)

    cv2.imwrite(str(output_path), image)


def main() -> None:
    args = parse_args()
    dataset_info = build_demo_dataset(force_rebuild=args.force_data)
    cfg = build_config(args.epochs, args.decoder)

    if args.task == 'prepare':
        print(f'Demo keypoint dataset is ready in: {json_path}')
        print(f'Train annotations: {dataset_info["train_ann"]}')
        print(f'Val annotations: {dataset_info["val_ann"]}')
        return

    checkpoint_path: Path | None = None
    if args.task in ('all', 'train'):
        checkpoint_path = train_pose_model(cfg)
        print(f'Training finished. Latest checkpoint: {checkpoint_path}')

    if args.task in ('all', 'infer'):
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        elif checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(WORK_DIR)

        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(
                'No checkpoint available for inference. Run training first or pass --checkpoint.')

        prediction_json = run_inference_on_all_samples(
            cfg,
            checkpoint_path,
            score_thr=args.score_thr,
            save_heatmaps=args.save_heatmaps)
        print(f'Inference summary saved to: {prediction_json}')


if __name__ == '__main__':
    main()
