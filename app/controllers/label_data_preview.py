from __future__ import annotations

import contextlib
import io
import json
import re
from pathlib import Path

from app.constants import IMAGE_SUFFIXES, LabelUsage, current_model_task
from app.controllers.base import TabController
from app.label_scan import (
    LabelStats,
    collect_label_shapes,
    detect_polygon_anomalies,
    detect_rectangle_labels,
    extract_json_labels,
    format_label_type_text,
    get_dominant_polygon_points,
    collect_point_labels,
)
from app.qt_imports import (
    Qt,
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from app.task_runner import run_thread_with_process_events
from app.theme import (
    label_table_combo_stylesheet,
    label_table_line_edit_stylesheet,
    label_table_stylesheet,
    message_box_stylesheet,
)
from s1_dataJson2Train import (
    ConvertInfo,
    compute_split_plan,
    precheck_convert,
    process_filesHRNet,
    process_filesYoloFeaturePoint,
    process_filesYoloObb,
    process_filesYoloSeg,
    write_pose_schema,
)

# 数据已经标注好了，划分训练集和验证集，预览标签映射。将标注数据写入到yolo,hrnet格式
class LabelDataPreviewController(TabController):
    def __init__(self, window):
        super().__init__(window)
        self.saved_label_mapping_rows = []
        self.label_id_edits = {}
        self.label_usage_combos = {}
        self.label_table = None
        self.keypoint_order_table = None
        self.saved_keypoint_order_rows = []

    def configure_ui(self):
        if not hasattr(self.window, "labelMapContainer") or self.window.labelMapContainer is None:
            self.label_table = None
            return

        self.window.labelMapContainer.setMinimumHeight(220)
        table_ss = label_table_stylesheet()

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
        self.label_table.setStyleSheet(table_ss)

        # 新格式/旧格式单选按钮（仅 Pose 任务显示，OBB/SEG 下隐藏）
        radio_new = QRadioButton("新格式（矩形=对象，点=关键点）")
        radio_new.setObjectName("radioPoseFormatNew")
        radio_legacy = QRadioButton("旧格式（polygon/line/point 按原有方式处理）")
        radio_legacy.setObjectName("radioPoseFormatLegacy")
        radio_legacy.setChecked(True)
        self.window.radioPoseFormatNew = radio_new
        self.window.radioPoseFormatLegacy = radio_legacy

        radio_layout = QHBoxLayout()
        radio_layout.setContentsMargins(0, 0, 0, 0)
        radio_layout.addWidget(radio_new)
        radio_layout.addWidget(radio_legacy)
        radio_layout.addStretch()

        # 关键点排序表（本阶段只做结构，不做扫描填充）
        self.keypoint_order_table = QTableWidget(self.window.labelMapContainer)
        self.keypoint_order_table.setColumnCount(2)
        self.keypoint_order_table.setHorizontalHeaderLabels(["Point Label", "关键点序号"])
        self.keypoint_order_table.setAlternatingRowColors(True)
        self.keypoint_order_table.verticalHeader().setVisible(False)
        self.keypoint_order_table.horizontalHeader().setStretchLastSection(False)
        for column in range(2):
            self.keypoint_order_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
        self.keypoint_order_table.setMinimumHeight(220)
        self.keypoint_order_table.setStyleSheet(table_ss)

        keypoint_layout = QVBoxLayout()
        keypoint_layout.setContentsMargins(0, 0, 0, 0)
        keypoint_layout.addWidget(self.keypoint_order_table)
        keypoint_container = QWidget(self.window.labelMapContainer)
        keypoint_container.setLayout(keypoint_layout)

        # 对象类别映射表 / 关键点排序表
        self.window.keypointTabWidget = QTabWidget(self.window.labelMapContainer)
        self.window.keypointTabWidget.setObjectName("keypointTabWidget")
        self.window.keypointTabWidget.addTab(self.label_table, "对象类别映射表")
        self.window.keypointTabWidget.addTab(keypoint_container, "关键点排序表")
        self.window.keypointTabWidget.setTabBarAutoHide(True)

        outer_layout = QVBoxLayout(self.window.labelMapContainer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addLayout(radio_layout)
        outer_layout.addWidget(self.window.keypointTabWidget)

        self.window.labelTable = self.label_table
        self._populate_label_mapping_layout([])

    def sync_task_ui(self):
        """根据顶部任务单选按钮同步标签处理区域的可见性。

        仅 Pose 任务显示新/旧格式单选按钮与「关键点排序表」Tab；
        OBB / SEG 任务下两者隐藏，仅保留「对象类别映射表」。
        """
        is_pose = current_model_task(self.window) == "pose"
        for radio_name in ("radioPoseFormatNew", "radioPoseFormatLegacy"):
            radio = getattr(self.window, radio_name, None)
            if radio is not None:
                radio.setVisible(is_pose)
        tab_widget = getattr(self.window, "keypointTabWidget", None)
        if tab_widget is not None and tab_widget.count() > 1:
            try:
                tab_widget.setTabVisible(1, is_pose)
            except AttributeError:
                pass

    def _insert_keypoint_order_row(self):
        """Insert a single keypoint order row and register its spin widget."""
        if self.keypoint_order_table is None:
            return
        row = self.keypoint_order_table.rowCount()
        self.keypoint_order_table.insertRow(row)
        self.keypoint_order_table.setItem(row, 0, QTableWidgetItem(""))
        slot_item = QTableWidgetItem("0")
        slot_item.setTextAlignment(Qt.AlignCenter)
        self.keypoint_order_table.setItem(row, 1, slot_item)

    def populate_keypoint_order_from_scan(self, point_labels):
        """Reset-style populate: rebuild keypoint order table from scanned labels.

        This clears previous rows and suggestions, then inserts rows for each
        scanned label. All suggested slots are derived from label names and
        any manual adjustments are not preserved.
        """
        if self.keypoint_order_table is None:
            return
        labels = sorted({str(label).strip() for label in point_labels if str(label).strip()})
        if not labels:
            return

        used_slots = set()
        self.keypoint_order_table.blockSignals(True)
        # clear existing rows
        self.keypoint_order_table.setRowCount(0)

        for label_text in labels:
            self._insert_keypoint_order_row()
            last_row = self.keypoint_order_table.rowCount() - 1
            self.keypoint_order_table.setItem(last_row, 0, QTableWidgetItem(label_text))
            slot = self._suggest_slot_from_label(label_text, used_slots)
            slot_item = QTableWidgetItem(str(slot))
            slot_item.setTextAlignment(Qt.AlignCenter)
            self.keypoint_order_table.setItem(last_row, 1, slot_item)
            used_slots.add(slot)

        self.keypoint_order_table.blockSignals(False)

    @staticmethod
    def _suggest_slot_from_label(label_text, used_slots):
        """从标签名字中提取数字建议槽位（如 cap_01 -> 1）；不覆盖用户手改。"""
        numbers = re.findall(r"\d+", str(label_text))
        if numbers:
            candidate = int(numbers[-1])
            if 0 <= candidate <= 199 and candidate not in used_slots:
                return candidate
        slot = 0
        while slot in used_slots:
            slot += 1
        return slot

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
        point_labels = set()
        is_new_pose_format = (
            current_model_task(self.window) == "pose"
            and self.get_pose_format() == "rectangle_point"
        )

        for json_file in json_files:
            try:
                with json_file.open("r", encoding="utf-8-sig") as file:
                    payload = json.load(file)
            except (OSError, json.JSONDecodeError) as exc:
                error_messages.append(json_file.name)
                self.append_log(f"解析 JSON 失败: {json_file} ({exc})")
                continue

            if is_new_pose_format:
                # 新格式（矩形=对象，点=关键点）：对象类别只看 rectangle，关键点只看 point
                rectangle_labels = detect_rectangle_labels(payload)
                labels.update(rectangle_labels)
                point_labels.update(collect_point_labels(payload))
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
            update_nfp=not is_new_pose_format,
        )

        if is_new_pose_format:
            self.populate_keypoint_order_from_scan(point_labels)

        scan_detail = "新格式(rectangle_point)" if is_new_pose_format else "对象类别"
        self.append_log(
            f"扫描完成: {len(all_files)} 个文件，其中 {len(json_files)} 个 JSON 文件，"
            f"找到 {len(labels)} 个{scan_detail}标签，{len(point_labels)} 个关键点标签。"
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
        use_pose = not use_seg and not use_obb

        def _run_split_process():
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                    # 共享一次 train/val/test 划分，YOLO 与 HRNet 共用
                    split_plan = compute_split_plan(convert_info)
                    # 预检查：写入/删除 datasets 之前完成全部校验，任意错误即停止
                    precheck_convert(convert_info, split_plan)
                    if use_seg:
                        train_files, val_files, test_files = process_filesYoloSeg(convert_info, split_plan)
                    elif use_obb:
                        train_files, val_files, test_files = process_filesYoloObb(convert_info, split_plan)
                    else:
                        train_files, val_files, test_files = process_filesYoloFeaturePoint(convert_info, split_plan)
                        process_filesHRNet(convert_info, split_plan)
                        if convert_info.PoseFormat == "rectangle_point":
                            write_pose_schema(convert_info)
                result_holder["train_files"] = train_files
                result_holder["val_files"] = val_files
                result_holder["test_files"] = test_files
            except Exception as exc:
                result_holder["error"] = exc
            finally:
                result_holder["stdout"] = buffer.getvalue()

        task_name = "分割" if use_seg else ("obb" if use_obb else "关键点")
        pose_format_text = (
            f", PoseFormat={convert_info.PoseFormat}" if use_pose else ""
        )
        self.append_log(
            f"开始划分数据({task_name}): "
            f"Append={convert_info.Append}, TrainRatio={convert_info.TrainRatio}, "
            f"ValRatio={convert_info.ValRatio}, TestRatio={convert_info.TestRatio}, "
            f"Seed={convert_info.Seed}, NFP={convert_info.NFP}{pose_format_text}"
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

    def get_pose_format(self):
        """返回当前姿态数据格式: 'rectangle_point'（新格式）或 'legacy'（旧格式）。"""
        radio_new = getattr(self.window, "radioPoseFormatNew", None)
        if radio_new is not None and radio_new.isChecked():
            return "rectangle_point"
        return "legacy"

    def collect_keypoint_order_rows(self):
        """收集关键点排序表内容（跳过 label 为空的行）。"""
        rows = []
        if self.keypoint_order_table is None:
            return rows
        for row in range(self.keypoint_order_table.rowCount()):
            label_item = self.keypoint_order_table.item(row, 0)
            label_text = "" if label_item is None else label_item.text().strip()
            if not label_text:
                continue
            slot_item = self.keypoint_order_table.item(row, 1)
            try:
                slot = int(str(slot_item.text()).strip()) if slot_item is not None else 0
            except (TypeError, ValueError):
                slot = 0
            rows.append({"label": label_text, "slot": slot})
        return rows

    def restore_pose_format_state(self, pose_format, keypoint_rows):
        """恢复姿态格式单选与关键点排序表内容。"""
        radio_new = getattr(self.window, "radioPoseFormatNew", None)
        radio_legacy = getattr(self.window, "radioPoseFormatLegacy", None)
        is_new_format = pose_format == "rectangle_point"
        if radio_new is not None:
            radio_new.blockSignals(True)
            radio_new.setChecked(is_new_format)
            radio_new.blockSignals(False)
        if radio_legacy is not None:
            radio_legacy.blockSignals(True)
            radio_legacy.setChecked(not is_new_format)
            radio_legacy.blockSignals(False)
        self._populate_keypoint_order_rows(keypoint_rows)

    def _populate_keypoint_order_rows(self, rows):
        if self.keypoint_order_table is None:
            return
        valid_rows = [
            row
            for row in (rows or [])
            if isinstance(row, dict) and str(row.get("label", "")).strip()
        ]
        self.saved_keypoint_order_rows = valid_rows

        self.keypoint_order_table.blockSignals(True)
        self.keypoint_order_table.setRowCount(0)
        for row_payload in valid_rows:
            self._insert_keypoint_order_row()
            last_row = self.keypoint_order_table.rowCount() - 1
            label_text = str(row_payload.get("label", "")).strip()
            self.keypoint_order_table.setItem(last_row, 0, QTableWidgetItem(label_text))
            try:
                slot = max(0, min(199, int(row_payload.get("slot", 0))))
            except (TypeError, ValueError):
                slot = 0
            slot_item = QTableWidgetItem(str(slot))
            slot_item.setTextAlignment(Qt.AlignCenter)
            self.keypoint_order_table.setItem(last_row, 1, slot_item)
        self.keypoint_order_table.blockSignals(False)

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
        update_nfp=True,
    ):
        label_stats = label_stats or {}
        sorted_labels = sorted({label.strip() for label in labels if str(label).strip()})
        self._populate_label_mapping_layout(sorted_labels, label_stats=label_stats)

        if (
            update_nfp
            and hasattr(self.window, "lineEditNumFeaturePoints")
            and self.window.lineEditNumFeaturePoints is not None
        ):
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

        task = current_model_task(self.window)
        if task == "pose":
            convert_info.PoseFormat = self.get_pose_format()
            if convert_info.PoseFormat == "rectangle_point":
                keypoint_map = self._collect_keypoint_order_map()
                if keypoint_map is None:
                    return None
                # 界面槽位即内部槽位（0..N-1，与对象训练时ID保持一致）
                convert_info.KeypointOrder = dict(keypoint_map)
                convert_info.NFP = len(keypoint_map)
        else:
            convert_info.PoseFormat = "legacy"

        return convert_info

    def _collect_keypoint_order_map(self):
        """校验关键点排序表并返回 {label: slot}（slot 从 0 开始）。

        校验：至少一行、序号唯一、从 0 开始连续递增。失败返回 None 并提示。
        """
        rows = self.collect_keypoint_order_rows()
        if not rows:
            self.append_log("新格式需要先在「关键点排序表」中填写至少一行 Point Label。")
            return None
        labels = [row["label"] for row in rows]
        slots = [row["slot"] for row in rows]
        if len(set(labels)) != len(labels):
            self.append_log("关键点排序表中存在重复的 Point Label。")
            return None
        if len(set(slots)) != len(slots):
            self.append_log("关键点排序表中序号重复，每个序号只能使用一次。")
            return None
        if sorted(slots) != list(range(0, len(slots))):
            self.append_log("关键点排序表序号有空号，必须从 0 开始连续递增。")
            return None
        return dict(zip(labels, slots))

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
