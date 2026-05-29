from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, Set

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from HRNetHelper import DEFAULT_DECODER, build_config, inference_topdown, init_model, register_all_modules


BOX_SCORE_THR = 0.5
KEYPOINT_DRAW_THR = 0.3

# CLASS_ALIAS = {
#     0: 'FPC',
#     1: 'ZIF',
#     2: 'Line',
#     3: 'Camera',
# }

# GT_LABEL_ALIAS = {
#     '0': 'FPC',
#     '1': 'ZIF',
#     '2': 'Line',
#     '3': 'Camera',
#     'CameraFPC': 'FPC',
#     'Connector': 'ZIF',
#     'FPC': 'FPC',
#     'ZIF': 'ZIF',
# }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _resolve_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f'Image not found or unreadable: {image}')
        return img
    if isinstance(image, np.ndarray):
        return image
    raise TypeError(f'Unsupported image type: {type(image)!r}')


def _resolve_model_path(model_path: Optional[Union[str, Path]], *, subdir: str,default_file: str) -> Path:
    if model_path is None or str(model_path).strip() == '':
        model_path = Path('./runs') / subdir / default_file

    path = Path(model_path).expanduser()
    if path.exists():
        return path.resolve()

    last_checkpoint = Path('./runs') / subdir / str(model_path) / default_file
    return last_checkpoint.expanduser().resolve()


def _read_hrnet_checkpoint(last_checkpoint_path: Path) -> Path:
    if not last_checkpoint_path.exists():
        raise FileNotFoundError(f'last_checkpoint file not found: {last_checkpoint_path}')
    checkpoint_text = last_checkpoint_path.read_text(encoding='utf-8').strip()
    checkpoint_path = Path(checkpoint_text).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'checkpoint file not found: {checkpoint_path}')
    return checkpoint_path.resolve()


def _result_image_path(ret: Sequence[Any]) -> str:
    if not ret:
        return ''
    result = ret[0]
    for attr in ('path', 'img_path'):
        value = getattr(result, attr, None)
        if value:
            return str(value)
    return ''


def _result_image_size(image: Union[str, Path, np.ndarray, None], ret: Sequence[Any]) -> tuple[int, int]:
    if isinstance(image, np.ndarray):
        return int(image.shape[0]), int(image.shape[1])
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is not None:
            return int(img.shape[0]), int(img.shape[1])
    if ret:
        result = ret[0]
        orig_shape = getattr(result, 'orig_shape', None)
        if orig_shape is not None and len(orig_shape) >= 2:
            return int(orig_shape[0]), int(orig_shape[1])
    return 0, 0


def draw_results(image: Union[str, Path, np.ndarray], ret: Sequence[Any], class_names: Optional[Dict[int, str]] = None) -> np.ndarray:
    vis_img = _resolve_image(image).copy()
    if not ret:
        return vis_img

    result = ret[0]
    boxes_obj = getattr(result, 'boxes', None)
    keypoints_obj = getattr(result, 'keypoints', None)
    if boxes_obj is None or keypoints_obj is None:
        return vis_img

    boxes_xyxy = _to_numpy(boxes_obj.xyxy) if hasattr(boxes_obj, 'xyxy') else None
    boxes_conf = _to_numpy(boxes_obj.conf) if hasattr(boxes_obj, 'conf') else None
    boxes_cls = _to_numpy(boxes_obj.cls) if hasattr(boxes_obj, 'cls') else None
    keypoints_data = _to_numpy(keypoints_obj.data)

    try:
        keypoint_scores = _to_numpy(keypoints_obj.conf) if keypoints_obj.conf is not None else None
    except Exception:
        keypoint_scores = keypoints_data[:, :, 2] if keypoints_data.ndim == 3 and keypoints_data.shape[-1] >= 3 else None

    mapping = class_names or {}
    for obj_idx, keypoints in enumerate(keypoints_data):
        if boxes_conf is not None and obj_idx < len(boxes_conf) and float(boxes_conf[obj_idx]) < BOX_SCORE_THR:
            continue

        class_id = int(boxes_cls[obj_idx]) if boxes_cls is not None and obj_idx < len(boxes_cls) else -1
        label_name = mapping.get(class_id, str(class_id))

        if boxes_xyxy is not None and obj_idx < len(boxes_xyxy):
            x1, y1, x2, y2 = map(int, boxes_xyxy[obj_idx])
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            box_text = label_name
            if boxes_conf is not None and obj_idx < len(boxes_conf):
                box_text = f'{label_name} {float(boxes_conf[obj_idx]):.2f}'
            cv2.putText(vis_img, box_text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        pts = []
        for kp_idx, (x, y, *_) in enumerate(keypoints):
            score = 1.0
            if keypoint_scores is not None:
                score = float(keypoint_scores[obj_idx][kp_idx])
            if score < KEYPOINT_DRAW_THR:
                continue

            color = (0, 0, 255)
            if kp_idx == 0:
                color = (255, 0, 0)
            elif kp_idx == 1:
                color = (0, 255, 255)
            cv2.circle(vis_img, (int(x), int(y)), 4, color, -1)
            cv2.putText(vis_img, f'{kp_idx + 1}', (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            pts.append([int(x), int(y)])

        if len(pts) > 1:
            poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [poly], isClosed=True, color=(0, 255, 0), thickness=1)

    return vis_img


def save_result(image: Union[str, Path, np.ndarray, None], imgName: str, ret: Sequence[Any], save_path: Union[str, Path], class_names: Optional[Dict[int, str]] = None, part_labels: Optional[Set[int]] = None) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if part_labels is None:
        json_data: Dict[str, Any] = {
            'version': '5.0.1',
            'flags': {},
            'shapes': [],
            'imagePath': imgName,
            'imageData': None,
            'imageHeight': 0,
            'imageWidth': 0,
        }
    else:
        # 尝试在 image 所在路径或使用 imgName 将后缀替换为 .json 来找到同名的 json
        json_data = None
        candidates = None
        p = Path(image)
        candidates = p.with_suffix('.json')

        try:
            if candidates.exists():
                json_text = candidates.read_text(encoding='utf-8')
                json_data = json.loads(json_text)
        except Exception:
            # 忽略读取或解析错误，尝试下一个候选
            json_data = None

        if json_data is None:
            # 若未找到有效的 json，退回到新建结构
            json_data = {
                'version': '5.0.1',
                'flags': {},
                'shapes': [],
                'imagePath': imgName,
                'imageData': None,
                'imageHeight': 0,
                'imageWidth': 0,
            }

    img_h, img_w = _result_image_size(image, ret)
    json_data['imageHeight'] = img_h
    json_data['imageWidth'] = img_w

    if ret:
        result = ret[0]
        boxes_obj = getattr(result, 'boxes', None)
        keypoints_obj = getattr(result, 'keypoints', None)
        if boxes_obj is not None and keypoints_obj is not None:
            boxes_conf = _to_numpy(boxes_obj.conf) if hasattr(boxes_obj, 'conf') else None
            boxes_cls = _to_numpy(boxes_obj.cls) if hasattr(boxes_obj, 'cls') else None
            keypoints_data = _to_numpy(keypoints_obj.data)

            mapping = class_names or {}
            # part_labels 是一个可选的 int 集合，若提供则只保存这些 class_id 的条目
            part_set = set(part_labels) if part_labels is not None else None
            for obj_idx in range(keypoints_data.shape[0]):
                if boxes_conf is not None and obj_idx < len(boxes_conf) and float(boxes_conf[obj_idx]) < BOX_SCORE_THR:
                    continue
                class_id = int(boxes_cls[obj_idx]) if boxes_cls is not None and obj_idx < len(boxes_cls) else -1
                if part_set is not None and class_id not in part_set:
                    continue
                if class_id not in mapping:
                    continue

                json_data['shapes'].append({
                    'label': mapping[class_id],
                    'points': [[float(x), float(y)] for x, y, *_ in keypoints_data[obj_idx]],
                    'group_id': None,
                    'shape_type': 'polygon',
                    'flags': {},
                })

    with save_path.open('w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)


def statistics_result(gtFiles: Sequence[Union[str, Path, None]], predRet, class_names, workspace) -> None:
    from math import ceil
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd

    workspace_path = Path(workspace).expanduser().resolve()
    hist_dir = workspace_path / 'error_hist'
    hist_dir.mkdir(parents=True, exist_ok=True)

    class_id_to_name = {int(class_id): str(class_name) for class_id, class_name in class_names.items()}
    name_to_class_id = {str(class_name): int(class_id) for class_id, class_name in class_names.items()}
    class_order = [class_id_to_name[class_id] for class_id in sorted(class_id_to_name)]
    line_class_id = name_to_class_id.get('Line', 2)

    def _get_result_item(pred_item):
        if pred_item is None:
            return None
        if isinstance(pred_item, (list, tuple)):
            return pred_item[0] if pred_item else None
        return pred_item

    def _get_image_shape(result_item, gt_item):
        if result_item is not None:
            orig_shape = getattr(result_item, 'orig_shape', None)
            if orig_shape is not None and len(orig_shape) >= 2:
                return int(orig_shape[0]), int(orig_shape[1])

        if isinstance(gt_item, (str, Path)):
            gt_path = Path(gt_item)
            for image_dir_name in ('images', 'image', 'imgs', 'img'):
                candidate_dir = gt_path.parent.parent / image_dir_name / gt_path.parent.name
                if not candidate_dir.exists():
                    continue
                for suffix in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
                    candidate = candidate_dir / f'{gt_path.stem}{suffix}'
                    if candidate.exists():
                        img = cv2.imread(str(candidate))
                        if img is not None:
                            return int(img.shape[0]), int(img.shape[1])
        return 0, 0

    def _load_gt_content(gt_item):
        if gt_item is None:
            return ''
        if isinstance(gt_item, (str, Path)):
            gt_path = Path(gt_item)
            if gt_path.exists():
                return gt_path.read_text(encoding='utf-8', errors='ignore')
            return str(gt_item)
        return str(gt_item)

    def _bbox_iou(box1, box2):
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h
        area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
        area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
        union = area1 + area2 - inter_area
        return (inter_area / union) if union > 0 else 0.0

    def _parse_gt_objects(gt_item, img_h, img_w):
        gt_content = _load_gt_content(gt_item)
        gt_objects = []
        if not gt_content:
            return gt_objects

        for raw_line in gt_content.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            tokens = line.split()
            if len(tokens) < 5:
                continue

            label_token = tokens[0]
            if label_token.isdigit() or (label_token.startswith('-') and label_token[1:].isdigit()):
                class_id = int(label_token)
            else:
                class_id = name_to_class_id.get(label_token)
                if class_id is None:
                    continue

            try:
                values = [float(token) for token in tokens[1:]]
            except ValueError:
                continue

            if len(values) < 4:
                continue

            cx, cy, bw, bh = values[:4]
            kp_values = values[4:]
            keypoints = []
            keypoint_scores = []
            for idx in range(0, len(kp_values) - 2, 3):
                x_norm, y_norm, visibility = kp_values[idx:idx + 3]
                keypoints.append([x_norm * img_w, y_norm * img_h])
                keypoint_scores.append(int(round(visibility)))

            x1 = max(0.0, (cx - bw / 2.0) * img_w)
            y1 = max(0.0, (cy - bh / 2.0) * img_h)
            x2 = min(float(img_w - 1), (cx + bw / 2.0) * img_w)
            y2 = min(float(img_h - 1), (cy + bh / 2.0) * img_h)

            gt_objects.append({
                'class_id': class_id,
                'class_name': class_id_to_name.get(class_id, str(class_id)),
                'box': np.array([x1, y1, x2, y2], dtype=np.float32),
                'keypoints': np.asarray(keypoints, dtype=np.float32),
                'keypoint_scores': np.asarray(keypoint_scores, dtype=np.int32),
            })
        return gt_objects

    def _build_pred_objects(pred_item):
        result_item = _get_result_item(pred_item)
        if result_item is None:
            return None, [], None

        boxes_obj = getattr(result_item, 'boxes', None)
        keypoints_obj = getattr(result_item, 'keypoints', None)
        if boxes_obj is None or keypoints_obj is None:
            return result_item, [], None

        boxes_conf = _to_numpy(boxes_obj.conf) if hasattr(boxes_obj, 'conf') else None
        boxes_cls = _to_numpy(boxes_obj.cls) if hasattr(boxes_obj, 'cls') else None
        boxes_xyxy = _to_numpy(boxes_obj.xyxy) if hasattr(boxes_obj, 'xyxy') else None
        keypoints_data = _to_numpy(keypoints_obj.data)

        try:
            keypoint_scores = _to_numpy(keypoints_obj.conf) if keypoints_obj.conf is not None else None
        except Exception:
            keypoint_scores = keypoints_data[:, :, 2] if keypoints_data.ndim == 3 and keypoints_data.shape[-1] >= 3 else None

        pred_objects = []
        for obj_idx in range(keypoints_data.shape[0]):
            if boxes_conf is not None and obj_idx < len(boxes_conf) and float(boxes_conf[obj_idx]) < BOX_SCORE_THR:
                continue

            class_id = int(boxes_cls[obj_idx]) if boxes_cls is not None and obj_idx < len(boxes_cls) else -1
            class_name = class_id_to_name.get(class_id, str(class_id))
            pred_box = boxes_xyxy[obj_idx] if boxes_xyxy is not None and obj_idx < len(boxes_xyxy) else None
            if pred_box is None:
                continue

            pred_objects.append({
                'index': obj_idx,
                'class_id': class_id,
                'class_name': class_name,
                'box': pred_box,
                'keypoints': keypoints_data[obj_idx, :, :2],
                'box_conf': float(boxes_conf[obj_idx]) if boxes_conf is not None and obj_idx < len(boxes_conf) else 1.0,
                'kp_conf': keypoint_scores[obj_idx] if keypoint_scores is not None and len(keypoint_scores) > obj_idx else None,
            })

        return result_item, pred_objects, keypoint_scores

    def _plot_hist_on_ax(ax, data, title, bins=30, color='#4C72B0'):
        if data:
            mean_val = np.mean(data)
            median_val = np.median(data)
            p995_val = np.percentile(data, 99.5)
            std_val = np.std(data)
            ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor='white')
            ax.axvline(mean_val, color='crimson', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.4f}')
            ax.axvline(median_val, color='#2CA02C', linestyle='-.', linewidth=1.5, label=f'Median: {median_val:.4f}')
            ax.axvline(p995_val, color='#9467BD', linestyle=':', linewidth=1.8, label=f'P99.5: {p995_val:.4f}')
            ax.legend()
            ax.set_title(f'{title}\nStd: {std_val:.4f}')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)
        ax.set_xlabel('Keypoint Error (pixel)')
        ax.set_ylabel('Count')

    dist_counts_by_class = {class_name: {} for class_name in class_order}
    all_dist_by_class = {class_name: [] for class_name in class_order}
    stats = {class_name: {'gt': 0, 'pred': 0, 'detected': 0} for class_name in class_order}
    image_error_rows = []
    max_error = -1.0
    max_error_img = ''

    total_items = min(len(gtFiles), len(predRet))
    if len(gtFiles) != len(predRet):
        print(f'[statistics_result] gtFiles/predRet length mismatch: {len(gtFiles)} vs {len(predRet)}; using first {total_items} pairs.')

    for idx in range(total_items):
        gt_item = gtFiles[idx]
        pred_item = predRet[idx]
        result_item, pred_objects, _ = _build_pred_objects(pred_item)
        img_h, img_w = _get_image_shape(result_item, gt_item)
        if img_h <= 0 or img_w <= 0:
            print(f'[statistics_result] skip sample {idx} because image size is unavailable.')
            continue

        gt_objects = _parse_gt_objects(gt_item, img_h, img_w)
        if not gt_objects:
            continue

        per_image_class_dists = {class_name: [] for class_name in class_order}
        used_pred_indices = set()
        pred_count_by_class = {class_name: 0 for class_name in class_order}
        for pred_obj in pred_objects:
            if pred_obj['class_name'] in pred_count_by_class:
                pred_count_by_class[pred_obj['class_name']] += 1

        for class_name, pred_count in pred_count_by_class.items():
            stats[class_name]['pred'] += pred_count

        for gt_obj in gt_objects:
            class_name = gt_obj['class_name']
            if class_name not in stats:
                continue

            stats[class_name]['gt'] += 1

            class_preds = [pred_obj for pred_obj in pred_objects if pred_obj['class_name'] == class_name and pred_obj['index'] not in used_pred_indices]
            if not class_preds:
                continue

            best_pred = None
            best_iou = -1.0
            for pred_obj in class_preds:
                iou = _bbox_iou(pred_obj['box'], gt_obj['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_obj

            if best_pred is None or best_iou < 0.3:
                continue

            used_pred_indices.add(best_pred['index'])
            stats[class_name]['detected'] += 1

            gt_points = gt_obj['keypoints']
            pred_points = np.asarray(best_pred['keypoints'], dtype=np.float32)
            if gt_points.size == 0 or pred_points.size == 0:
                continue

            if gt_obj['class_id'] != line_class_id and len(gt_points) != len(pred_points):
                continue

            compare_count = min(len(gt_points), len(pred_points))
            kp_conf = best_pred['kp_conf']
            valid_dists = []
            for kp_idx in range(compare_count):
                if gt_obj['keypoint_scores'].size > kp_idx and int(gt_obj['keypoint_scores'][kp_idx]) == 0:
                    continue
                if kp_conf is not None:
                    kp_score = float(kp_conf[kp_idx])
                    if kp_score < KEYPOINT_DRAW_THR:
                        continue

                dist = float(np.linalg.norm(gt_points[kp_idx] - pred_points[kp_idx]))
                valid_dists.append(dist)
                per_image_class_dists[class_name].append(dist)
                all_dist_by_class[class_name].append(dist)
                dist_counts_by_class.setdefault(class_name, {}).setdefault(kp_idx + 1, []).append(dist)

            if valid_dists:
                img_max_dist = max(valid_dists)
                if img_max_dist > max_error:
                    max_error = img_max_dist
                    max_error_img = str(gt_item) if isinstance(gt_item, (str, Path)) else str(idx)

        image_row = {'image': str(gt_item) if isinstance(gt_item, (str, Path)) else str(idx)}
        for class_name in class_order:
            image_row[f'{class_name}_mean'] = float(np.mean(per_image_class_dists[class_name])) if per_image_class_dists[class_name] else np.nan
        image_error_rows.append(image_row)

    if not any(values for values in all_dist_by_class.values()):
        print('[statistics_result] no valid keypoint distances were collected.')
        return

    def _save_histogram(data, title, save_path, bins=30):
        plt.figure(figsize=(8, 5))
        if data:
            mean_val = np.mean(data)
            median_val = np.median(data)
            p995_val = np.percentile(data, 99.5)
            std_val = np.std(data)
            plt.hist(data, bins=bins, color='#4C72B0', alpha=0.85, edgecolor='white')
            plt.axvline(mean_val, color='crimson', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.4f}')
            plt.axvline(median_val, color='#2CA02C', linestyle='-.', linewidth=1.5, label=f'Median: {median_val:.4f}')
            plt.axvline(p995_val, color='#9467BD', linestyle=':', linewidth=1.8, label=f'P99.5: {p995_val:.4f}')
            plt.legend()
            plt.title(f'{title}\nStd: {std_val:.4f}')
        else:
            plt.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title(title)
        plt.xlabel('Keypoint Error (pixel)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    for class_name in class_order:
        dist_counts = dist_counts_by_class.get(class_name, {})
        keypoint_ids = sorted(dist_counts.keys())
        if not keypoint_ids:
            continue

        ncols = 2 if len(keypoint_ids) > 1 else 1
        nrows = ceil(len(keypoint_ids) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax_idx, kp_id in enumerate(keypoint_ids):
            ax = axes[ax_idx]
            dists = dist_counts[kp_id]
            if dists:
                mean_val = np.mean(dists)
                median_val = np.median(dists)
                p995_val = np.percentile(dists, 99.5)
                std_val = np.std(dists)
                ax.hist(dists, bins=30, color='#55A868', alpha=0.85, edgecolor='white')
                ax.axvline(mean_val, color='crimson', linestyle='--', linewidth=1.2, label=f'Mean: {mean_val:.4f}')
                ax.axvline(median_val, color='#2CA02C', linestyle='-.', linewidth=1.2, label=f'Median: {median_val:.4f}')
                ax.axvline(p995_val, color='#9467BD', linestyle=':', linewidth=1.4, label=f'P99.5: {p995_val:.4f}')
                ax.legend()
                ax.set_title(f'KP {kp_id} (n={len(dists)})\nStd: {std_val:.4f}')
                ax.set_xlabel('Error (pixel)')
                ax.set_ylabel('Count')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
                ax.set_title(f'KP {kp_id} (n=0)')
                ax.set_xticks([])
                ax.set_yticks([])

        for ax in axes[len(keypoint_ids):]:
            ax.axis('off')

        fig.suptitle(f'{class_name} Keypoint Error Histogram by Keypoint', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(hist_dir / f'{class_name.lower()}_hist_by_keypoint.png', dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, class_name in zip(axes, class_order):
        _plot_hist_on_ax(
            ax,
            all_dist_by_class[class_name],
            f"{class_name} Combined Keypoint Error\n(n={len(all_dist_by_class[class_name])})",
            color='#4C72B0',
        )
    for ax in axes[len(class_order):]:
        ax.axis('off')
    fig.suptitle('Combined Keypoint Error Distribution', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(hist_dir / 'combined.png', dpi=200)
    plt.close(fig)

    for class_name in class_order:
        _save_histogram(
            all_dist_by_class[class_name],
            f'{class_name} Combined Keypoint Error Histogram',
            hist_dir / f'{class_name.lower()}_hist_combined.png',
        )

    csv_path = hist_dir / 'image_error_summary.csv'
    csv_columns = ['image'] + [f'{class_name}_mean' for class_name in class_order]
    pd.DataFrame(image_error_rows, columns=csv_columns).to_csv(csv_path, index=False)

    for class_name in class_order:
        dist_counts = dist_counts_by_class[class_name]
        print('\n' + '=' * 20 + f' {class_name} Keypoint Error Stats ' + '=' * 20)
        print('-' * 60)
        print(f"{'Keypoint':<10} | {'Avg':<8} | {'Min':<8} | {'Max':<8} | {'Std':<8} | {'Count':<6}")
        print('-' * 60)
        if not dist_counts:
            print('No data')
            continue
        for kp_id in sorted(dist_counts.keys()):
            dists = dist_counts[kp_id]
            if dists:
                dists_np = np.asarray(dists, dtype=np.float32)
                print(f"KP {kp_id:<7} | {np.mean(dists_np):<8.4f} | {np.min(dists_np):<8.4f} | {np.max(dists_np):<8.4f} | {np.std(dists_np):<8.4f} | {len(dists):<6}")
            else:
                print(f"KP {kp_id:<7} | {'N/A':<43}")
        print('-' * 60)

    if max_error_img:
        print(f'Max Error Image: {max_error_img} with error: {max_error:.4f}')
    print('-' * 50)

    print('\n' + '=' * 60)
    print(f"{'Class':<15} | {'GT (Manual)':<10} | {'Detected GT':<12} | {'Recall':<8} | {'Pred (All)':<10}")
    print('-' * 60)
    for class_name in class_order:
        stats_row = stats[class_name]
        recall = (stats_row['detected'] / stats_row['gt'] * 100) if stats_row['gt'] > 0 else 0.0
        print(f"{class_name:<15} | {stats_row['gt']:<10} | {stats_row['detected']:<12} | {recall:>6.2f}% | {stats_row['pred']:<10}")
    print('=' * 60 + '\n')
    print('-' * 50)
    print(f'Image error summary saved to: {csv_path}')
    print(f'Error histogram saved to: {hist_dir}')


class InferenceModel:

    def __init__(self, class_alias: Optional[Dict[int, str]] = None):
        self.yolo_model = None
        self.hrnet_model = None
        self.class_alias: Dict[int, str] = class_alias or {}

    def load_yolo_model(self, model_path: Optional[Union[str, Path]] = None) -> YOLO:
        self.yolo_model = YOLO(model_path)
        return self.yolo_model

    def set_class_alias(self, class_alias: Optional[Dict[int, str]] = None) -> None:
        """Set or update the class id -> name mapping."""
        self.class_alias = class_alias or {}

    def load_hrnet_model(self, model_path: Optional[Union[str, Path]] = None) -> Any:
        last_checkpoint_path = Path(f'{model_path}/last_checkpoint').expanduser().resolve()
        if not last_checkpoint_path.exists():
            raise FileNotFoundError(f'last_checkpoint file not found: {last_checkpoint_path}')

        checkpoint_text = last_checkpoint_path.read_text(encoding='utf-8').strip()
        checkpoint_path = Path(checkpoint_text).expanduser()


        cfg = build_config(300, DEFAULT_DECODER)
        register_all_modules(init_default_scope=True)
        self.hrnet_model = init_model(cfg, str(checkpoint_path), device='cuda:0')
        return self.hrnet_model

    def predict(self, image: Union[str, Path, np.ndarray], conf_thr: float = BOX_SCORE_THR) -> Sequence[Any]:
        if self.yolo_model is None:
            return None

        img = _resolve_image(image)
        ret = self.yolo_model.predict(img, verbose=False)
        if not ret or self.hrnet_model is None:
            return ret

        if self.hrnet_model is None:
            return None

        result = ret[0]
        boxes_obj = getattr(result, 'boxes', None)
        if boxes_obj is None or len(boxes_obj) == 0:
            return ret

        box_conf = _to_numpy(boxes_obj.conf)
        img_h, img_w = img.shape[:2]
        bboxes: List[List[float]] = []
        for xyxy in _to_numpy(boxes_obj.xyxy)[box_conf >= conf_thr]:
            x1, y1, x2, y2 = map(float, xyxy)
            x1 = max(0.0, x1 - 10.0)
            y1 = max(0.0, y1 - 10.0)
            x2 = min(float(img_w - 1), x2 + 10.0)
            y2 = min(float(img_h - 1), y2 + 10.0)
            bboxes.append([x1, y1, x2 - x1, y2 - y1])

        if not bboxes:
            return ret

        hrnet_results = inference_topdown(self.hrnet_model, img, bboxes=bboxes, bbox_format='xywh')
        if not hrnet_results:
            return ret

        hrnet_keypoints_list = []
        hrnet_scores_list = []
        for sample in hrnet_results:
            pred = sample.pred_instances
            kp = _to_numpy(pred.keypoints)
            if kp.ndim == 2:
                kp = kp[None, ...]
            hrnet_keypoints_list.append(kp)

            scores = None
            if hasattr(pred, 'keypoint_scores') and pred.keypoint_scores is not None:
                scores = _to_numpy(pred.keypoint_scores)
                if scores.ndim == 1:
                    scores = scores[None, ...]
            if scores is not None:
                hrnet_scores_list.append(scores)

        hrnet_keypoints = np.concatenate(hrnet_keypoints_list, axis=0) if hrnet_keypoints_list else None
        hrnet_scores = np.concatenate(hrnet_scores_list, axis=0) if hrnet_scores_list else None
        if hrnet_keypoints is None:
            return ret

        try:
            yolo_keypoints = result.keypoints.data.detach().cpu().numpy()
        except Exception:
            return ret
        if yolo_keypoints.ndim == 2:
            yolo_keypoints = yolo_keypoints[None, ...]

        n_instances = min(yolo_keypoints.shape[0], hrnet_keypoints.shape[0])
        updated_keypoints = yolo_keypoints.copy()
        updated_keypoints[:n_instances, :, :2] = hrnet_keypoints[:n_instances, :, :2]

        if hrnet_scores is not None:
            if updated_keypoints.shape[-1] < 3:
                updated_keypoints = np.concatenate(
                    [updated_keypoints, np.zeros((*updated_keypoints.shape[:2], 1), dtype=updated_keypoints.dtype)],
                    axis=-1,
                )
            updated_keypoints[:n_instances, :, 2] = hrnet_scores[:n_instances, :updated_keypoints.shape[1]]

        updated_keypoints = torch.as_tensor(updated_keypoints, dtype=torch.float32, device=result.keypoints.data.device)
        result.update(keypoints=updated_keypoints)
        return ret


def testModel():
    workspace = "/home/ebots/Desktop/zhq/VisualFactoryTest/"

    model = InferenceModel()
    model.load_yolo_model(workspace + 'runs/pose/train_20260528_130710/weights/best.pt')
    model.load_hrnet_model(workspace + 'runs/HRNet/train_20260528_130710')
    ret = model.predict('./2026-04-27-09_39_35_833677__Cam1__Frame0.png')
    vis = draw_results('./2026-04-27-09_39_35_833677__Cam1__Frame0.png', ret, class_names={0: 'socket', 1: 'FPC', 2: 'Line', 3: 'Camera'})
    cv2.imwrite('./inference_vis.png', vis)
    save_result('./2026-04-27-09_39_35_833677__Cam1__Frame0.png', '2026-04-27-09_39_35_833677__Cam1__Frame0.png', ret, './inference_result.json', class_names={0: 'socket', 1: 'FPC', 2: 'Line', 3: 'Camera'})

def test_err_hist():
    workspace = "/home/ebots/Desktop/zhq/VisualFactoryTest/"
    yoloModel = workspace + 'runs/pose/train_20260528_130710/weights/best.pt'
    hrnetModel = workspace + 'runs/HRNet/train_20260528_130710'
    model = InferenceModel()
    model.load_yolo_model(yoloModel)
    model.load_hrnet_model(hrnetModel)
    infer_path = Path(workspace) / 'group_data' / 'group_014'

    # supported image extensions
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    pred_ret = []
    gt_file = []

    # support single image file or directory
    if infer_path.is_file():
        files = [infer_path]
    else:
        files = [p for p in sorted(infer_path.rglob('*')) if p.is_file() and p.suffix.lower() in exts]

    for img_path in files:
        # ensure it's an image file
        if img_path.suffix.lower() not in exts:
            continue

        ret = model.predict(str(img_path))
        pred_ret.append(ret)

        # find corresponding gt .txt under datasets by image stem
        dataset_dir = Path(workspace) / 'datasets'
        stem = img_path.stem
        txt_path = None
        if dataset_dir.exists():
            p1 = dataset_dir / 'labels' / 'train' / f'{stem}.txt'
            p2 = dataset_dir / 'labels' / 'val' / f'{stem}.txt'
            p3 = dataset_dir / 'labels' / 'test' / f'{stem}.txt'
            if p1.exists():
                txt_path = p1
            elif p2.exists():
                txt_path = p2
            elif p3.exists():
                txt_path = p3
            else:
                txt_path = None

        if txt_path is None:
            gt_file.append(None)
        else:
            gt_file.append(txt_path)
    statistics_result(gt_file, pred_ret, class_names={0: 'socket', 1: 'FPC', 2: 'Line', 3: 'Camera'}, workspace=workspace)

if __name__ == '__main__':
    # testModel()
    test_err_hist()