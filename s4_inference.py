from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

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
        model_path = default_name

    path = Path(model_path).expanduser()
    if path.exists():
        return path.resolve()

    last_checkpoint = Path('./runs/HRNet') / str(model_path) / default_file
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


def save_result(image: Union[str, Path, np.ndarray, None], imgName: str, ret: Sequence[Any], save_path: Union[str, Path], class_names: Optional[Dict[int, str]] = None) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    json_data: Dict[str, Any] = {
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
            for obj_idx in range(keypoints_data.shape[0]):
                if boxes_conf is not None and obj_idx < len(boxes_conf) and float(boxes_conf[obj_idx]) < BOX_SCORE_THR:
                    continue

                class_id = int(boxes_cls[obj_idx]) if boxes_cls is not None and obj_idx < len(boxes_cls) else -1
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


def statistics_result(gt: Any, pred: Any) -> Dict[str, int]:
    return {
        'gt_count': len(gt) if gt is not None else 0,
        'pred_count': len(pred) if pred is not None else 0,
    }


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
            try:
                txt_content = txt_path.read_text(encoding='utf-8')
            except Exception:
                txt_content = txt_path.read_text(encoding='utf-8', errors='ignore')
            gt_file.append({'txt_path': str(txt_path), 'content': txt_content})
           

if __name__ == '__main__':
    testModel()
    # test_err_hist()