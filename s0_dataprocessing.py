import random
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def find_images(src: Path):
    for path in src.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            yield path


def _collect_images_system(source_folder: Path, selected_cams):
    """system：按 denali/相机目录收集图片。

    目录约定：
      denali/
        {ts0}/Cam0  → denali0_cam0
        {ts0}/Cam1  → denali0_cam1
        {ts1}/Cam0  → denali1_cam0
        ...
    denali 下子目录按名字排序后，第 0 个是 denali0，第 1 个是 denali1。
    """
    if not selected_cams:
        return []

    selected = set(selected_cams)
    result = []

    for denali_dir in source_folder.rglob('*'):
        if not denali_dir.is_dir() or denali_dir.name.lower() != 'denali':
            continue

        # denali 下的时间戳目录：排序后 index 对应 denali0 / denali1
        instance_dirs = sorted(
            (p for p in denali_dir.iterdir() if p.is_dir()),
            key=lambda p: p.name,
        )
        for denali_idx, instance_dir in enumerate(instance_dirs):
            for cam_dir in instance_dir.iterdir():
                if not cam_dir.is_dir():
                    continue
                cam_name = cam_dir.name.lower()  # Cam0 -> cam0
                if not cam_name.startswith('cam'):
                    continue
                key = f'denali{denali_idx}_{cam_name}'  # denali0_cam0
                if key in selected:
                    result.extend(find_images(cam_dir))

    return result


def _collect_images_unknow(source_folder: Path):
    """unknow：不区分来源，收集目录下全部图片。"""
    return list(find_images(source_folder))


def _collect_images_ui(source_folder: Path, selected_cams=None):
    """ui：数据源筛选与处理（占位）。"""
    print(f'ui source_mode is not implemented yet: {source_folder}')
    return []


def collect_images(source_folder: Path, source_mode: str = 'system', selected_cams=None):
    """按数据来源收集图片。

    source_mode:
      - system: 按 denali/相机目录筛选（依赖 selected_cams）
      - unknow: 全部图片混在一起
      - ui: 预留扩展
    """
    source_folder = Path(source_folder)
    if source_mode == 'system':
        return _collect_images_system(source_folder, selected_cams)
    if source_mode == 'unknow':
        return _collect_images_unknow(source_folder)
    if source_mode == 'ui':
        return _collect_images_ui(source_folder, selected_cams=selected_cams)
    raise ValueError(f"Unsupported source_mode: {source_mode!r}")


def chunked(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _existing_group_start_id(output_folder: Path):
    if not output_folder.exists():
        return 0

    max_group_id = -1
    for child in output_folder.iterdir():
        if not child.is_dir() or not child.name.startswith('group_'):
            continue

        suffix = child.name[len('group_'):]
        if suffix.isdigit():
            max_group_id = max(max_group_id, int(suffix))

    return max_group_id + 1


def _normalize_image_to_uint8(img):
    if img.dtype == np.uint8:
        return img

    MIN = img.min()
    MAX = img.max()

    img = (img.astype(np.float32) - MIN) / (MAX - MIN) * 255

    return img.astype(np.uint8)


def make_copy_groups(images, source_folder: Path, out_dir: Path, group_size: int, start_id: int = 0,
                    tiff2png: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    created_groups = 0

    for index, group in enumerate(chunked(images, group_size)):
        grp_dir = out_dir / f'group_{index + start_id:03d}'
        grp_dir.mkdir(exist_ok=True)
        created_groups += 1

        for path in group:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f'Failed to read image: {path}')
                continue
            if tiff2png:
                img = _normalize_image_to_uint8(img)

            rel_path = path.relative_to(source_folder)
            renamed = '__'.join(rel_path.with_suffix('.png').parts)
            dest = grp_dir / renamed
            if not dest.exists():
                cv2.imwrite(str(dest), img)

    return created_groups


def process_data(source_folder, output_folder, group_size, tiff2png: bool = False,
                 selected_cams=None, source_mode: str = 'system'):
    source_folder = Path(source_folder).expanduser()
    output_folder = Path(output_folder).expanduser()

    if not source_folder.exists() or not source_folder.is_dir():
        raise FileNotFoundError(f"Source folder '{source_folder}' does not exist or is not a directory.")

    if group_size < 1:
        raise ValueError('group_size must be greater than 0')

    output_folder.mkdir(parents=True, exist_ok=True)

    images = collect_images(
        source_folder,
        source_mode=source_mode,
        selected_cams=selected_cams,
    )

    if not images:
        print('No images found in', source_folder)
        return {
            'source_folder': str(source_folder),
            'output_folder': str(output_folder),
            'image_count': 0,
            'group_count': 0,
            'source_mode': source_mode,
        }

    random.shuffle(images)

    print(f'Found {len(images)} images in {source_folder} — grouping every {group_size} images')

    start_id = _existing_group_start_id(output_folder)
    group_count = make_copy_groups(
        images,
        source_folder,
        output_folder,
        group_size,
        start_id=start_id,
        tiff2png=tiff2png,
    )

    return {
        'source_folder': str(source_folder),
        'output_folder': str(output_folder),
        'image_count': len(images),
        'group_count': group_count,
        'source_mode': source_mode,
    }

if __name__ == '__main__':
    process_data(r'C:\Users\eBots\Desktop\XiaomiProject\origin_data\failure_images_20260509',
                 r'C:\Users\eBots\Desktop\XiaomiProject\group_data',
                 100)
