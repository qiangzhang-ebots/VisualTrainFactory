from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from app.constants import IMAGE_SUFFIXES


@dataclass
class LabelStats:
    types: Dict[str, int] = field(default_factory=dict)
    polygon_max_points: int = 0
    polygon_point_histogram: Dict[int, int] = field(default_factory=dict)

    def merge(self, other: "LabelStats") -> None:
        for shape_type, count in other.types.items():
            self.types[shape_type] = self.types.get(shape_type, 0) + count
        self.polygon_max_points = max(self.polygon_max_points, other.polygon_max_points)
        for points_count, count in other.polygon_point_histogram.items():
            self.polygon_point_histogram[points_count] = (
                self.polygon_point_histogram.get(points_count, 0) + count
            )

    def to_dict(self) -> dict:
        return {
            "types": dict(self.types),
            "polygon_max_points": self.polygon_max_points,
            "polygon_point_histogram": dict(self.polygon_point_histogram),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LabelStats":
        if not isinstance(payload, dict):
            return cls()
        histogram = payload.get("polygon_point_histogram") or {}
        normalized_histogram = {}
        for key, value in histogram.items():
            try:
                normalized_histogram[int(key)] = int(value)
            except (TypeError, ValueError):
                continue
        types = payload.get("types") or {}
        return cls(
            types={str(key): int(value) for key, value in types.items()},
            polygon_max_points=int(payload.get("polygon_max_points") or 0),
            polygon_point_histogram=normalized_histogram,
        )


def _normalize_text(value) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return ""


def _points_count_from_value(value) -> int:
    if not isinstance(value, list):
        return 0
    return sum(1 for item in value if isinstance(item, (list, tuple)) and len(item) >= 2)


def extract_json_labels(payload) -> Tuple[Set[str], Dict[str, LabelStats], List[Tuple[str, int]]]:
    labels: Set[str] = set()
    label_stats: Dict[str, LabelStats] = {}
    polygon_entries: List[Tuple[str, int]] = []
    stack = [payload]
    interesting_keys = {"label", "labels", "class", "class_name", "category", "category_name", "name"}
    shape_type_keys = {"shape_type", "type", "geometry", "geometry_type"}

    def _update_stats(label_text: str, shape_type_text=None, polygon_points_count=0):
        entry = label_stats.setdefault(label_text, LabelStats())
        if shape_type_text:
            entry.types[shape_type_text] = entry.types.get(shape_type_text, 0) + 1
            if shape_type_text == "polygon" and polygon_points_count > entry.polygon_max_points:
                entry.polygon_max_points = polygon_points_count
            if shape_type_text == "polygon" and polygon_points_count > 0:
                entry.polygon_point_histogram[polygon_points_count] = (
                    entry.polygon_point_histogram.get(polygon_points_count, 0) + 1
                )

    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            dict_label_candidates = []
            dict_shape_type = None
            dict_polygon_points_count = 0

            for shape_key in shape_type_keys:
                shape_value = current.get(shape_key)
                normalized_shape = _normalize_text(shape_value)
                if normalized_shape:
                    dict_shape_type = normalized_shape.lower()
                    break

            if dict_shape_type == "polygon":
                dict_polygon_points_count = _points_count_from_value(current.get("points"))

            for key, value in current.items():
                if key in interesting_keys:
                    if isinstance(value, str):
                        normalized = _normalize_text(value)
                        if normalized:
                            labels.add(normalized)
                            dict_label_candidates.append(normalized)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                normalized = _normalize_text(item)
                                if normalized:
                                    labels.add(normalized)
                                    dict_label_candidates.append(normalized)
                            else:
                                stack.append(item)

                if isinstance(value, (dict, list)):
                    stack.append(value)

            for label_text in dict_label_candidates:
                _update_stats(label_text, dict_shape_type, dict_polygon_points_count)
                if dict_shape_type == "polygon" and dict_polygon_points_count > 0:
                    polygon_entries.append((label_text, dict_polygon_points_count))
        elif isinstance(current, list):
            stack.extend(current)

    return labels, label_stats, polygon_entries


def format_label_type_text(stats: LabelStats) -> str:
    if not stats.types:
        return ""

    primary_type = max(stats.types.items(), key=lambda item: (item[1], item[0]))[0]
    if primary_type == "polygon":
        if stats.polygon_point_histogram:
            dominant_points, dominant_count = max(
                stats.polygon_point_histogram.items(),
                key=lambda item: (item[1], item[0]),
            )
            if dominant_count > 0:
                return f"polygon-{dominant_points}"
        if stats.polygon_max_points > 0:
            return f"polygon-{stats.polygon_max_points}"
    return primary_type


def get_dominant_polygon_points(label_stats: Dict[str, LabelStats]) -> str:
    if not label_stats:
        return ""

    point_label_counts = {}
    for stats in label_stats.values():
        if not stats.polygon_point_histogram:
            continue
        dominant_points, _ = max(
            stats.polygon_point_histogram.items(),
            key=lambda item: (item[1], item[0]),
        )
        point_label_counts[dominant_points] = point_label_counts.get(dominant_points, 0) + 1

    if not point_label_counts:
        return ""

    return str(max(point_label_counts.items(), key=lambda item: (item[1], item[0]))[0])


def find_related_image_paths(json_file_path) -> List[Path]:
    json_path = Path(json_file_path)
    related_images = []
    for suffix in IMAGE_SUFFIXES:
        candidate = json_path.with_suffix(suffix)
        if candidate.exists() and candidate.is_file():
            related_images.append(candidate)
    return related_images


def detect_polygon_anomalies(
    polygon_records: Iterable[dict],
    polygon_point_histogram: Dict[str, Dict[int, int]],
) -> List[str]:
    anomaly_logs = []
    for label_text, points_hist in polygon_point_histogram.items():
        if len(points_hist) <= 1:
            continue

        majority_points, majority_count = max(points_hist.items(), key=lambda item: (item[1], item[0]))
        if majority_count <= 1:
            continue

        for record in polygon_records:
            if record["label"] != label_text:
                continue
            if record["points"] == majority_points:
                continue

            related_images = find_related_image_paths(record["json_file"])
            target_files = related_images if related_images else [record["json_file"]]
            target_text = ", ".join(str(path) for path in target_files)
            anomaly_logs.append(
                f"异常图片: label={label_text}, 主流点数={majority_points}, "
                f"当前点数={record['points']}, 文件={target_text}"
            )
    return anomaly_logs
