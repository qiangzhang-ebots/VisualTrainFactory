from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

from app.constants import IMAGE_SUFFIXES, LabelUsage, current_model_task
from app.controllers.base import TabController
from app.label_scan import (
    LabelStats,
    detect_polygon_anomalies,
    extract_json_labels,
    format_label_type_text,
    get_dominant_polygon_points,
)
from app.qt_imports import (
    QAbstractItemView,
    QComboBox,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)
from app.task_runner import run_thread_with_process_events
from app.theme import label_table_combo_stylesheet, label_table_line_edit_stylesheet, message_box_stylesheet
from s1_dataJson2Train import (
    ConvertInfo,
    process_filesHRNet,
    process_filesYoloFeaturePoint,
    process_filesYoloObb,
    process_filesYoloSeg,
)

# 数据已经标注好了，划分训练集和验证集，预览标签映射。将标注数据写入到yolo,hrnet格式
class LabelDataPreviewController(TabController):
    def __init__(self, window):
        super().__init__(window)
        self.saved_label_mapping_rows = []
        self.label_id_edits = {}
        self.label_usage_combos = {}
        self.label_table = None

    def configure_ui(self):
        if not hasattr(self.window, "labelMapContainer") or self.window.labelMapContainer is None:
            self.label_table = None
            return

        self.window.labelMapContainer.setMinimumHeight(220)
        self.label_table = QTableWidget(self.window.labelMapContainer)
        self.label_table.setColumnCount(4)
        self.label_table.setHorizontalHeaderLabels(["Label", "类型", "训练时ID", "请选择用途"])
        self.label_table.setAlternatingRowColors(True)
        self.label_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.label_table.verticalHeader().setVisible(False)
        self.label_table.horizontalHeader().setStretchLastSection(False)
        for column in range(4):
            self.label_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
        self.label_table.setMinimumHeight(220)

        outer_layout = QVBoxLayout(self.window.labelMapContainer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.label_table)

        self.window.labelTable = self.label_table
        self._populate_label_mapping_layout([])

    def on_enter(self):
        self.append_log('已切换到数据划分及预览页。可以点击"扫描所有文件"刷新 group_data 标签。')

    def scan_group_data_labels(self):
        workspace = self.get_workspace()
        if workspace is None:
            self.append_log("请先选择有效的工作目录。")
            self.refresh_preview_label_scan_result([])
            return

        group_data_dir = workspace.group_data
        if not group_data_dir.exists():
            self.append_log(f"未找到 group_data 目录: {group_data_dir}")
            self.refresh_preview_label_scan_result([], error_messages=[])
            return

        all_files = [path for path in group_data_dir.rglob("*") if path.is_file()]
        json_files = sorted(path for path in all_files if path.suffix.lower() == ".json")
        if not json_files:
            self.append_log(f"在 {group_data_dir} 中没有找到 JSON 文件。")
            self.refresh_preview_label_scan_result(
                [], scanned_file_count=len(all_files), json_file_count=0
            )
            return

        labels = set()
        label_stats = {}
        polygon_point_histogram = {}
        polygon_records = []
        error_messages = []

        for json_file in json_files:
            try:
                with json_file.open("r", encoding="utf-8-sig") as file:
                    payload = json.load(file)
            except (OSError, json.JSONDecodeError) as exc:
                error_messages.append(json_file.name)
                self.append_log(f"解析 JSON 失败: {json_file} ({exc})")
                continue

            extracted_labels, extracted_stats, polygon_entries = extract_json_labels(payload)
            labels.update(extracted_labels)
            for label_text, stats in extracted_stats.items():
                current_stats = label_stats.setdefault(label_text, LabelStats())
                current_stats.merge(stats)

            for label_text, points_count in polygon_entries:
                label_hist = polygon_point_histogram.setdefault(label_text, {})
                label_hist[points_count] = label_hist.get(points_count, 0) + 1
                polygon_records.append(
                    {"label": label_text, "points": points_count, "json_file": json_file}
                )

        anomaly_logs = detect_polygon_anomalies(polygon_records, polygon_point_histogram)
        if anomaly_logs:
            self.append_log("检测到 polygon 点数异常:")
            for message in anomaly_logs:
                self.append_log(message)

        stats_dict = {key: value.to_dict() for key, value in label_stats.items()}
        self.refresh_preview_label_scan_result(
            labels,
            scanned_file_count=len(all_files),
            json_file_count=len(json_files),
            error_messages=error_messages,
            label_stats=stats_dict,
        )
        self.append_log(
            f"扫描完成: {len(all_files)} 个文件，其中 {len(json_files)} 个 JSON 文件，找到 {len(labels)} 个标签。"
        )

    def run_train_data_split(self):
        workspace = self.get_workspace()
        if workspace is None:
            self.append_log("请先选择有效的工作目录。")
            return

        group_data_dir = workspace.group_data
        self.append_log(f"当前数据划分预览目录: {group_data_dir}")
        if not group_data_dir.exists():
            self.append_log("group_data 目录还不存在，请先生成数据。")
            return

        json_file_count = sum(1 for _ in group_data_dir.rglob("*.json"))
        image_file_count = sum(
            1
            for path in group_data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        self.append_log(f"group_data 统计: {image_file_count} 张图片，{json_file_count} 个 JSON 文件。")

        convert_info = self._build_train_split_convert_info(group_data_dir)
        if convert_info is None:
            return

        used_ids = sorted(set(convert_info.Label2Int.values()))
        if used_ids:
            expected_ids = list(range(len(used_ids)))
            if used_ids != expected_ids:
                error_msg = (
                    "标签ID不符合YOLO要求！\n\n"
                    "YOLO要求标签ID必须从0开始连续递增（0, 1, 2, 3...）。\n\n"
                    f"当前标签ID: {used_ids}\n"
                    f"期望标签ID: {expected_ids}\n\n"
                    '请修正标签映射中的"训练时ID"后再进行数据划分。'
                )
                self.append_log(error_msg.replace("\n", " "))
                msg_box = QMessageBox(self.window)
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setWindowTitle("标签ID检查失败")
                msg_box.setText(error_msg)
                msg_box.setStyleSheet(message_box_stylesheet())
                msg_box.exec()
                return

        result_holder = {
            "error": None,
            "stdout": "",
            "train_files": None,
            "val_files": None,
            "test_files": None,
        }

        task = current_model_task(self.window)
        use_obb = task == "obb"
        use_seg = task == "seg"

        def _run_split_process():
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                    if use_seg:
                        train_files, val_files, test_files = process_filesYoloSeg(convert_info)
                    elif use_obb:
                        train_files, val_files, test_files = process_filesYoloObb(convert_info)
                    else:
                        train_files, val_files, test_files = process_filesYoloFeaturePoint(convert_info)
                        process_filesHRNet(convert_info)
                result_holder["train_files"] = train_files
                result_holder["val_files"] = val_files
                result_holder["test_files"] = test_files
            except Exception as exc:
                result_holder["error"] = exc
            finally:
                result_holder["stdout"] = buffer.getvalue()

        task_name = "分割" if use_seg else ("obb" if use_obb else "关键点")
        self.append_log(
            f"开始划分数据({task_name}): "
            f"Append={convert_info.Append}, TrainRatio={convert_info.TrainRatio}, "
            f"ValRatio={convert_info.ValRatio}, TestRatio={convert_info.TestRatio}, "
            f"Seed={convert_info.Seed}, NFP={convert_info.NFP}"
        )

        run_thread_with_process_events(_run_split_process)

        if result_holder["stdout"]:
            for line in result_holder["stdout"].splitlines():
                if line.strip():
                    self.append_log(line)

        if result_holder["error"] is not None:
            self.append_log(f"数据划分失败: {result_holder['error']}")
            return

        train_files = result_holder["train_files"] or []
        val_files = result_holder["val_files"] or []
        test_files = result_holder["test_files"] or []
        self.append_log(
            "数据划分完成: "
            f"train={len(train_files)}，val={len(val_files)}，test={len(test_files)}，"
            f"输出目录: {convert_info.DatasetsDir}"
        )
        self.window.mark_image_list_dirty()

    def collect_label_mapping_rows(self):
        rows = []
        for row_index, (label_text, line_edit) in enumerate(self.label_id_edits.items()):
            usage_combo = self.label_usage_combos.get(label_text)
            type_item = None
            if self.label_table is not None:
                type_item = self.label_table.item(row_index, 1)

            rows.append(
                {
                    "label": label_text,
                    "train_id": "" if line_edit is None else line_edit.text().strip(),
                    "usage": "" if usage_combo is None else usage_combo.currentText().strip(),
                    "type_text": "" if type_item is None else type_item.text().strip(),
                }
            )
        return rows

    def restore_label_mapping_rows(self, rows):
        if not isinstance(rows, list):
            rows = []

        self.saved_label_mapping_rows = [
            row for row in rows if isinstance(row, dict) and str(row.get("label", "")).strip()
        ]
        label_names = [str(row["label"]).strip() for row in self.saved_label_mapping_rows]
        if not label_names:
            return
        self.refresh_preview_label_scan_result(label_names, label_stats={})

    def get_label_id_mapping(self):
        mapping = {}
        for label_text, line_edit in self.label_id_edits.items():
            if line_edit is None:
                continue
            value = line_edit.text().strip()
            if value:
                mapping[label_text] = int(value)
        return mapping

    def refresh_preview_label_scan_result(
        self,
        labels,
        scanned_file_count=0,
        json_file_count=0,
        error_messages=None,
        label_stats=None,
    ):
        label_stats = label_stats or {}
        sorted_labels = sorted({label.strip() for label in labels if str(label).strip()})
        self._populate_label_mapping_layout(sorted_labels, label_stats=label_stats)

        if hasattr(self.window, "lineEditNumFeaturePoints") and self.window.lineEditNumFeaturePoints is not None:
            parsed_stats = {
                key: LabelStats.from_dict(value) if isinstance(value, dict) else value
                for key, value in label_stats.items()
            }
            self.window.lineEditNumFeaturePoints.blockSignals(True)
            self.window.lineEditNumFeaturePoints.setText(get_dominant_polygon_points(parsed_stats))
            self.window.lineEditNumFeaturePoints.blockSignals(False)

        self.window.inference_controller.populate_save_part_label_widget()

    def _populate_label_mapping_layout(self, labels, label_stats=None):
        if self.label_table is None:
            return

        label_stats = label_stats or {}
        previous_values = {}
        for label_text, line_edit in self.label_id_edits.items():
            if line_edit is not None:
                previous_values[label_text] = line_edit.text().strip()

        for row in self.saved_label_mapping_rows:
            label_text = str(row.get("label", "")).strip()
            if label_text:
                previous_values.setdefault(label_text, str(row.get("train_id", "")).strip())

        previous_usages = {}
        for label_text, combo_box in self.label_usage_combos.items():
            if combo_box is not None:
                previous_usages[label_text] = combo_box.currentText()

        for row in self.saved_label_mapping_rows:
            label_text = str(row.get("label", "")).strip()
            if label_text:
                previous_usages.setdefault(label_text, str(row.get("usage", "")).strip())

        previous_type_texts = {}
        for row in self.saved_label_mapping_rows:
            label_text = str(row.get("label", "")).strip()
            if label_text:
                previous_type_texts.setdefault(label_text, str(row.get("type_text", "")).strip())

        self.label_id_edits = {}
        self.label_usage_combos = {}
        self.label_table.clearContents()
        self.label_table.clearSpans()

        if not labels:
            self.label_table.setRowCount(1)
            hint_item = QTableWidgetItem("扫描后将在这里显示每个 label 的类型和 id 映射。")
            self.label_table.setItem(0, 0, hint_item)
            self.label_table.setSpan(0, 0, 1, 4)
            return

        self.label_table.setRowCount(len(labels))
        for row_index, label_text in enumerate(labels):
            stats_payload = label_stats.get(label_text, {})
            stats = (
                LabelStats.from_dict(stats_payload)
                if isinstance(stats_payload, dict)
                else stats_payload
            )
            type_text = format_label_type_text(stats) or previous_type_texts.get(label_text, "")

            self.label_table.setItem(row_index, 0, QTableWidgetItem(label_text))
            self.label_table.setItem(row_index, 1, QTableWidgetItem(type_text))

            row_input = QLineEdit(self.label_table)
            row_input.setPlaceholderText("输入训练时对应的 id")
            row_input.setStyleSheet(label_table_line_edit_stylesheet())
            default_id_text = self.default_id_text_from_label(label_text)
            row_input.setText(previous_values.get(label_text, default_id_text))
            self.label_table.setCellWidget(row_index, 2, row_input)

            usage_combo = QComboBox(self.label_table)
            usage_combo.addItems(
                [LabelUsage.UNSET.value, LabelUsage.TRAIN.value, LabelUsage.OCCLUDED.value]
            )
            previous_usage = str(previous_usages.get(label_text, "")).strip()
            if previous_usage in ("", LabelUsage.UNSET.value):
                previous_usage = LabelUsage.TRAIN.value
            usage_combo.setCurrentText(previous_usage)
            usage_combo.setStyleSheet(label_table_combo_stylesheet())
            self.label_table.setCellWidget(row_index, 3, usage_combo)

            self.label_id_edits[label_text] = row_input
            self.label_usage_combos[label_text] = usage_combo

    def _build_train_split_convert_info(self, group_data_directory: Path):
        workspace = self.get_workspace()
        if workspace is None:
            self.append_log("请先选择有效的工作目录。")
            return None

        convert_info = ConvertInfo()
        convert_info.JsonPath = str(group_data_directory)
        convert_info.DatasetsDir = str(workspace.datasets)

        write_mode_combo = getattr(self.window, "comboBoxPreviewWriteMode", None)
        convert_info.Append = bool(write_mode_combo is not None and write_mode_combo.currentIndex() == 1)

        train_ratio_widget = getattr(self.window, "doubleSpinBoxPreviewTrainRatio", None)
        val_ratio_widget = getattr(self.window, "doubleSpinBoxPreviewValRatio", None)
        test_ratio_widget = getattr(self.window, "doubleSpinBoxPreviewTestRatio", None)
        seed_widget = getattr(self.window, "spinBoxPreviewSeed", None)
        nfp_widget = getattr(self.window, "lineEditNumFeaturePoints", None)

        if train_ratio_widget is not None:
            convert_info.TrainRatio = float(train_ratio_widget.value())
        if val_ratio_widget is not None:
            convert_info.ValRatio = float(val_ratio_widget.value())
        if test_ratio_widget is not None:
            convert_info.TestRatio = float(test_ratio_widget.value())
        if seed_widget is not None:
            convert_info.Seed = int(seed_widget.value())

        if nfp_widget is not None:
            try:
                convert_info.NFP = max(1, int(str(nfp_widget.text()).strip()))
            except (TypeError, ValueError):
                self.append_log("NFP 输入无效，已回退到默认值 4。")

        convert_info.Label2Int = self._collect_label2int_mapping()
        convert_info.OccupiedLabel = self.collect_occluded_labels()
        if not convert_info.Label2Int:
            self.append_log('没有读取到可用的标签映射，请先扫描并设置"训练时ID"。')
            return None

        return convert_info

    def _collect_label2int_mapping(self):
        mapping = {}
        for label_text, line_edit in self.label_id_edits.items():
            if line_edit is None:
                continue

            usage_combo = self.label_usage_combos.get(label_text)
            if usage_combo is not None:
                usage_text = usage_combo.currentText().strip()
                if usage_text and usage_text != LabelUsage.TRAIN.value:
                    continue

            raw_value = line_edit.text().strip()
            if not raw_value:
                continue

            try:
                mapping[label_text] = int(raw_value)
            except ValueError:
                self.append_log(f"标签 {label_text} 的训练时ID无效，已忽略: {raw_value}")

        return mapping

    def collect_occluded_labels(self):
        occupied_labels = []
        for label_text, usage_combo in self.label_usage_combos.items():
            if usage_combo is None:
                continue
            usage_text = usage_combo.currentText().strip()
            if usage_text != LabelUsage.OCCLUDED.value:
                continue
            occupied_labels.append(label_text)
        return occupied_labels

    @staticmethod
    def default_id_text_from_label(label_text):
        try:
            return str(int(str(label_text).strip()))
        except (ValueError, TypeError):
            return ""
