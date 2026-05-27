import random
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def find_images(src: Path):
    for path in src.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            yield path


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

    max_val = img.max()
    if max_val <= 0:
        return img.astype(np.uint8)

    if max_val > 255:
        return (img / (max_val / 255.0)).astype(np.uint8)

    return img.astype(np.uint8)


def make_copy_groups(images, source_folder: Path, out_dir: Path, group_size: int, start_id: int = 0):
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

            img = _normalize_image_to_uint8(img)

            rel_path = path.relative_to(source_folder)
            renamed = '__'.join(rel_path.with_suffix('.png').parts)
            dest = grp_dir / renamed
            if not dest.exists():
                cv2.imwrite(str(dest), img)

    return created_groups


def process_data(source_folder, output_folder, group_size):
    source_folder = Path(source_folder).expanduser()
    output_folder = Path(output_folder).expanduser()

    if not source_folder.exists() or not source_folder.is_dir():
        raise FileNotFoundError(f"Source folder '{source_folder}' does not exist or is not a directory.")

    if group_size < 1:
        raise ValueError('group_size must be greater than 0')

    output_folder.mkdir(parents=True, exist_ok=True)

    images = list(find_images(source_folder))
    if not images:
        print('No images found in', source_folder)
        return {
            'source_folder': str(source_folder),
            'output_folder': str(output_folder),
            'image_count': 0,
            'group_count': 0,
        }

    random.shuffle(images)

    print(f'Found {len(images)} images in {source_folder} — grouping every {group_size} images')

    start_id = _existing_group_start_id(output_folder)
    group_count = make_copy_groups(images, source_folder, output_folder, group_size, start_id=start_id)

    return {
        'source_folder': str(source_folder),
        'output_folder': str(output_folder),
        'image_count': len(images),
        'group_count': group_count,
    }

if __name__ == '__main__':
    process_data(r'C:\Users\eBots\Desktop\XiaomiProject\origin_data\failure_images_20260509',
                 r'C:\Users\eBots\Desktop\XiaomiProject\group_data',
                 100)