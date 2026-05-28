import json
import os
import random
import shutil
from pathlib import Path


class ConvertInfo:
	def __init__(self):
		self.Append = False
		self.TrainRatio = 0.8
		self.ValRatio = 0.1
		self.TestRatio = 0.1
		self.NFP = 4
		self.Label2Int = {}
		self.OccupiedLabel = []
		self.JsonPath = './labeldata'
		self.DatasetsDir = './datasets'
		self.Seed = 42


# 兼容旧代码里可能写成 ConverInfo 的调用方式。
ConverInfo = ConvertInfo


IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def _script_dir():
	return Path(__file__).resolve().parent


def _resolve_path(path_text):
	path = Path(str(path_text))
	if path.is_absolute():
		return path
	return (_script_dir() / path).resolve()


def _normalize_label_map(label_map):
	result = {}
	for raw_label, raw_value in (label_map or {}).items():
		label_text = str(raw_label)
		try:
			label_id = int(raw_value)
		except (TypeError, ValueError) as exc:
			raise ValueError(f'Label2Int 中的值必须是整数: {raw_label} -> {raw_value}') from exc
		result[label_text] = label_id
	return result


def _get_occupied_labels(convert_info):
	return {str(label) for label in getattr(convert_info, 'OccupiedLabel', [])}


def _normalize_rect(rect):
	if not isinstance(rect, (list, tuple)):
		raise ValueError(f'OccupiedLabel 对应的 rectangle 必须是列表或元组: {rect}')

	if len(rect) == 4 and all(isinstance(value, (int, float)) for value in rect):
		left, top, right, bottom = rect
		return float(min(left, right)), float(min(top, bottom)), float(max(left, right)), float(max(top, bottom))

	if len(rect) == 2 and all(isinstance(point, (list, tuple)) and len(point) == 2 for point in rect):
		(x1, y1), (x2, y2) = rect
		return float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))

	raise ValueError(f'OccupiedLabel 对应的 rectangle 只支持 [left, top, right, bottom] 或 [[x1, y1], [x2, y2]]: {rect}')


def _point_in_rect(x, y, rect):
	left, top, right, bottom = rect
	return left <= x <= right and top <= y <= bottom


def _collect_occupied_rectangles(shapes, json_file, occupied_labels):
	occupied_rects = []
	for shape in shapes:
		label_text = str(shape.get('label', ''))
		if label_text not in occupied_labels:
			continue
		if shape.get('shape_type', '') != 'rectangle':
			continue
		points = shape.get('points', [])
		if len(points) != 2:
			raise RuntimeError(f'遮挡矩形必须包含2个点: {json_file} 中的 shape {shape}')
		occupied_rects.append(_normalize_rect(points))
	return occupied_rects


def _resolve_point_visibility(x, y, occupied_rects):
	for rect in occupied_rects:
		if _point_in_rect(x, y, rect):
			return 0
	return 2


def _to_yolo_visibility(visible):
	if visible == 1:
		return 2
	return visible


def _shape_points_to_keypoints(points, json_file, nfp, occupied_rects):
	if not 1 <= len(points) <= nfp:
		raise RuntimeError(f'不支持的点数量: {json_file} 中的 shape_points={points}')

	keypoints = [
		(
			float(x),
			float(y),
			_resolve_point_visibility(float(x), float(y), occupied_rects),
		)
		for x, y in points
	]
	keypoints.extend([(0.0, 0.0, 0)] * (nfp - len(keypoints)))
	return keypoints


def _resolve_category_id(label_text, convert_info):
	label_map = _normalize_label_map(convert_info.Label2Int)
	if label_text in label_map:
		return label_map[label_text]
	if label_text.isdigit():
		return int(label_text)
	raise RuntimeError(f'不支持的类别标签: {label_text}')


def _flatten_rel_path(rel_path):
	rel_path = str(rel_path)
	return rel_path.replace('\\', '_').replace('/', '_')


def _collect_json_files(json_root):
	all_files = []
	for root_dir, _, files in os.walk(json_root):
		for file_name in files:
			if file_name.lower().endswith('.json'):
				rel_path = os.path.relpath(os.path.join(root_dir, file_name), json_root)
				all_files.append(rel_path)
	all_files.sort()
	return all_files


def _collect_existing_stems(datasets_dir):
	existing_files = set()
	if not datasets_dir.exists():
		return existing_files

	for split in ('train', 'val', 'test'):
		for folder in ('images', 'labels'):
			target_dir = datasets_dir / folder / split
			if not target_dir.exists():
				continue
			for file_name in target_dir.iterdir():
				if file_name.is_file():
					existing_files.add(file_name.stem)
	return existing_files


def _split_file_list(all_files, convert_info, existing_stems=None):
	existing_stems = existing_stems or set()
	selected_files = []
	for json_name in all_files:
		flat_stem = _flatten_rel_path(os.path.splitext(json_name)[0])
		if flat_stem not in existing_stems:
			selected_files.append(json_name)

	if convert_info.Append and existing_stems and not selected_files:
		print('append模式：没有发现新增数据')
		return [], [], []

	if not convert_info.Append:
		selected_files = all_files[:]

	random.seed(getattr(convert_info, 'Seed', 42))
	random.shuffle(selected_files)

	total = len(selected_files)
	if total == 0:
		return [], [], []

	ratio_values = [float(convert_info.TrainRatio), float(convert_info.ValRatio), float(convert_info.TestRatio)]
	if sum(ratio_values) <= 0:
		raise ValueError('TrainRatio / ValRatio / TestRatio 之和必须大于 0')
	ratio_sum = sum(ratio_values)
	ratio_values = [value / ratio_sum for value in ratio_values]

	train_end = int(total * ratio_values[0])
	val_end = train_end + int(total * ratio_values[1])
	train_files = selected_files[:train_end]
	val_files = selected_files[train_end:val_end]
	test_files = selected_files[val_end:]
	return train_files, val_files, test_files


def _ensure_dataset_dirs(datasets_dir, include_test=True):
	if datasets_dir.exists():
		return

	for split in ('train', 'val'):
		(datasets_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
		(datasets_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
	if include_test:
		(datasets_dir / 'images' / 'test').mkdir(parents=True, exist_ok=True)
		(datasets_dir / 'labels' / 'test').mkdir(parents=True, exist_ok=True)


def _prepare_output_dirs(datasets_dir, convert_info):
	if not convert_info.Append:
		shutil.rmtree(datasets_dir, ignore_errors=True)
	_include_test = float(convert_info.TestRatio) > 0
	_ensure_dataset_dirs(datasets_dir, include_test=_include_test)


def _find_image_file(json_root, rel_json_path, data):
	rel_base = Path(rel_json_path).with_suffix('')
	for suffix in IMAGE_SUFFIXES:
		candidate = json_root / f'{rel_base.as_posix()}{suffix}'
		if candidate.exists():
			return candidate, suffix

	image_path = data.get('imagePath')
	if image_path:
		candidate = Path(image_path)
		if candidate.is_absolute() and candidate.exists():
			return candidate, candidate.suffix.lower()
		candidate = (json_root / image_path).resolve()
		if candidate.exists():
			return candidate, candidate.suffix.lower()

	raise FileNotFoundError(f'图片文件不存在: {json_root / rel_base}')


def _get_image_size(image_file):
	try:
		from PIL import Image
	except ImportError as exc:
		raise ImportError('需要安装 Pillow 才能读取图片尺寸') from exc

	with Image.open(image_file) as image:
		return image.size


def build_yolo_pose_line(label, keypoints, img_w, img_h):
	visible_points = [point for point in keypoints if point[2] != 0]
	if label == 2:
		box_points = visible_points if visible_points else keypoints
	else:
		box_points = keypoints

	xs = [point[0] for point in box_points]
	ys = [point[1] for point in box_points]
	min_x = max(min(xs) - 10, 0)
	max_x = min(max(xs) + 10, img_w)
	min_y = max(min(ys) - 10, 0)
	max_y = min(max(ys) + 10, img_h)
	cx = (min_x + max_x) / 2 / img_w
	cy = (min_y + max_y) / 2 / img_h
	w = (max_x - min_x) / img_w
	h = (max_y - min_y) / img_h

	parts = [f'{label}', f'{cx:.6f}', f'{cy:.6f}', f'{w:.6f}', f'{h:.6f}']
	for x, y, visible in keypoints:
		visible = _to_yolo_visibility(visible)
		if visible == 0:
			x, y = 0, 0
		parts.extend([f'{x / img_w:.6f}', f'{y / img_h:.6f}', str(visible)])
	return ' '.join(parts)


def quad_shape_to_yolo_line(shape, json_file, img_w, img_h, convert_info, occupied_rects):
	label_text = str(shape.get('label', ''))
	label = _resolve_category_id(label_text, convert_info)
	points = shape.get('points', [])
	if len(points) not in (1, 2, 4):
		raise RuntimeError(f'四边形/线段/单点标注必须包含1个、2个或4个点: {json_file} 中的 shape {shape}')
	keypoints = _shape_points_to_keypoints(points, json_file, convert_info.NFP, occupied_rects)
	return build_yolo_pose_line(label, keypoints, img_w, img_h)


def labelme2yolo(json_file, txt_file, img_w, img_h, convert_info):
	with open(json_file, 'r', encoding='utf-8') as file:
		data = json.load(file)

	lines = []
	shapes = data.get('shapes', [])
	occupied_labels = _get_occupied_labels(convert_info)
	occupied_rects = _collect_occupied_rectangles(shapes, json_file, occupied_labels)
	label_map = _normalize_label_map(convert_info.Label2Int)

	for shape in shapes:
		label_text = str(shape.get('label', ''))
		shape_type = shape.get('shape_type', '')
		if label_text in occupied_labels and shape_type == 'rectangle':
			continue

		if shape_type not in ('polygon', 'line', 'point'):
			continue

		points = shape.get('points', [])
		if not 1 <= len(points) <= convert_info.NFP:
			continue

		if label_text not in label_map and not label_text.isdigit():
			continue

		lines.append(quad_shape_to_yolo_line(shape, json_file, img_w, img_h, convert_info, occupied_rects))

	with open(txt_file, 'w', encoding='utf-8') as file:
		file.write('\n'.join(lines))


def _build_coco_categories(convert_info, files, occupied_labels):
	label_map = _normalize_label_map(convert_info.Label2Int)
	if label_map:
		items = sorted(label_map.items(), key=lambda item: item[1])
		return [
			{'id': label_id, 'name': label_text, 'supercategory': 'shape'}
			for label_text, label_id in items
			if label_text not in occupied_labels
		]

	label_ids = set()
	for rel_path in files:
		json_file = _resolve_path(Path(convert_info.JsonPath) / rel_path)
		with open(json_file, 'r', encoding='utf-8') as file:
			data = json.load(file)
		for shape in data.get('shapes', []):
			label_text = str(shape.get('label', ''))
			if label_text in occupied_labels:
				continue
			if label_text.isdigit():
				label_ids.add(int(label_text))
			elif label_text in label_map:
				label_ids.add(label_map[label_text])

	return [{'id': label_id, 'name': str(label_id), 'supercategory': 'shape'} for label_id in sorted(label_ids)]


def process_filesYolo(convert_info: ConvertInfo):
	json_root = _resolve_path(convert_info.JsonPath)
	datasets_dir = _resolve_path(convert_info.DatasetsDir)
	if not json_root.exists():
		raise FileNotFoundError(f'标注目录不存在: {json_root}')

	all_files = _collect_json_files(json_root)
	_prepare_output_dirs(datasets_dir, convert_info)
	existing_stems = _collect_existing_stems(datasets_dir) if convert_info.Append else set()
	train_files, val_files, test_files = _split_file_list(all_files, convert_info, existing_stems)

	for split in ('train', 'val', 'test'):
		(datasets_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
		(datasets_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

	for split_name, file_list in (('train', train_files), ('val', val_files), ('test', test_files)):
		for idx, json_name in enumerate(file_list):
			print('{split} {idx}/{total}: {json_name}'.format(split=split_name, idx=idx + 1, total=len(file_list), json_name=json_name))
			rel_json = Path(json_name)
			json_file = json_root / rel_json
			with open(json_file, 'r', encoding='utf-8') as file:
				data = json.load(file)

			image_file, image_suffix = _find_image_file(json_root, json_name, data)
			dst_name = rel_json.stem+image_suffix
			dst_img = datasets_dir / 'images' / split_name / dst_name
			shutil.copy(image_file, dst_img)

			img_w, img_h = _get_image_size(image_file)
			dst_txt = datasets_dir / 'labels' / split_name / f'{rel_json.stem}.txt'
			labelme2yolo(json_file, dst_txt, img_w, img_h, convert_info)

	return train_files, val_files, test_files


def _shape_to_coco_annotation(shape, json_file, image_id, annotation_id, image_width, image_height, convert_info, occupied_rects):
	label_text = str(shape.get('label', ''))
	if label_text in _get_occupied_labels(convert_info):
		return None

	if shape.get('shape_type', '') not in ('polygon', 'line', 'point'):
		return None

	points = shape.get('points', [])
	if not 1 <= len(points) <= convert_info.NFP:
		return None

	label_id = _resolve_category_id(label_text, convert_info)
	keypoints = _shape_points_to_keypoints(points, json_file, convert_info.NFP, occupied_rects)

	visible_points = [point for point in keypoints if point[2] != 0]
	bbox_points = visible_points if visible_points else keypoints
	xs = [int(round(point[0])) for point in bbox_points]
	ys = [int(round(point[1])) for point in bbox_points]
	min_x = max(min(xs) - 10, 0)
	max_x = min(max(xs) + 10, image_width)
	min_y = max(min(ys) - 10, 0)
	max_y = min(max(ys) + 10, image_height)
	bbox_width = max(max_x - min_x, 1)
	bbox_height = max(max_y - min_y, 1)

	flattened_keypoints = []
	for x, y, visible in keypoints:
		flattened_keypoints.extend([int(round(x)), int(round(y)), int(visible)])

	return {
		'id': annotation_id,
		'image_id': image_id,
		'category_id': label_id,
		'bbox': [min_x, min_y, bbox_width, bbox_height],
		'area': bbox_width * bbox_height,
		'iscrowd': 0,
		'num_keypoints': convert_info.NFP,
		'keypoints': flattened_keypoints,
	}


def process_filesHRNet(convert_info: ConvertInfo):
	json_root = _resolve_path(convert_info.JsonPath)
	datasets_dir = _resolve_path(convert_info.DatasetsDir)
	if not json_root.exists():
		raise FileNotFoundError(f'标注目录不存在: {json_root}')

	all_files = _collect_json_files(json_root)
	datasets_dir.mkdir(parents=True, exist_ok=True)
	existing_stems = _collect_existing_stems(datasets_dir) if convert_info.Append else set()
	train_files, val_files, test_files = _split_file_list(all_files, convert_info, existing_stems)

	occupied_labels = _get_occupied_labels(convert_info)
	categories = _build_coco_categories(convert_info, train_files + val_files + test_files, occupied_labels)

	for split_name, file_list in (('train', train_files), ('val', val_files), ('test', test_files)):
		dataset = {
			'images': [],
			'annotations': [],
			'categories': categories,
		}

		annotation_id = 1
		for image_id, rel_path in enumerate(file_list, start=1):
			json_file = json_root / rel_path
			with open(json_file, 'r', encoding='utf-8') as file:
				data = json.load(file)

			occupied_rects = _collect_occupied_rectangles(data.get('shapes', []), json_file, occupied_labels)

			image_width = int(data['imageWidth'])
			image_height = int(data['imageHeight'])
			image_file, image_suffix = _find_image_file(json_root, rel_path, data)
			file_name = _flatten_rel_path(Path(rel_path).with_suffix('')) + image_suffix

			dataset['images'].append(
				{
					'id': image_id,
					'file_name': file_name,
					'width': image_width,
					'height': image_height,
				}
			)

			for shape in data.get('shapes', []):
				annotation = _shape_to_coco_annotation(
					shape,
					json_file,
					image_id,
					annotation_id,
					image_width,
					image_height,
					convert_info,
					occupied_rects,
				)
				if annotation is None:
					continue
				dataset['annotations'].append(annotation)
				annotation_id += 1

		output_file = datasets_dir / f'{split_name}_keypoints.json'
		with open(output_file, 'w', encoding='utf-8') as file:
			json.dump(dataset, file, ensure_ascii=False, indent=2)

	return train_files, val_files, test_files


if __name__ == '__main__':
	workspace = "/home/ebots/Desktop/zhq/VisualFactoryTest/"
	convert_info = ConvertInfo()
	convert_info.Label2Int = {
		'1': 1,
		'2': 2,
	}
	convert_info.OccupiedLabel = ['v0', 'v1']
	# convert_info.Append = True
	convert_info.JsonPath = workspace + 'group_data'
	convert_info.DatasetsDir = workspace + 'datasets'

	try:
		process_filesYolo(convert_info)
		process_filesHRNet(convert_info)
	except FileNotFoundError as exc:
		print(exc)
