from __future__ import annotations

import json
import threading
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import QModelIndex, QTimer
from PyQt5.QtWidgets import QFileDialog

from app.constants import (
    CONFIG_FILE,
    CONFIG_VERSION,
    MAX_RECENT_WORK_DIRECTORIES,
    RECENT_WORK_DIRECTORY_KEY,
    WORK_DIRECTORY_CONFIG_NAME,
)
from app.controllers import (
    DataPrepareController,
    InferenceController,
    LabelDataPreviewController,
    TrainController,
    TrainDataVisualizationController,
)
from app.qt_imports import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QMainWindow,
    QRadioButton,
    QSpinBox,
    QWidget,
    QEvent,
    Qt,
    Signal,
    QFileSystemModel,
)
from app.theme import apply_checkable_indicator_styles, apply_theme, work_directory_combo_stylesheet
from app.workspace import Workspace


class VisualTrainFactoryWindow(QMainWindow):
    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        ui_path = Path(__file__).resolve().parent.parent / "VisualTrainFactory.ui"
        uic.loadUi(str(ui_path), self)

        self.log_message.connect(self._append_log_message_ui)
        apply_checkable_indicator_styles(self)
        apply_theme(self)

        self._folder_tree_model = QFileSystemModel(self)
        self.recent_work_directories = self._load_recent_work_directories()
        self.saved_widget_state = {}
        self.image_pairs = []
        self.current_index = -1
        self.is_image_list_dirty = True
        self.image_view = None
        self.error_image_view = None

        # 数据准备页，原始数据，划分group，开始标注
        self.data_prepare_controller = DataPrepareController(self)
        # 数据已经标注好了，划分训练集和验证集，预览标签映射。将标注数据写入到yolo,hrnet格式
        self.label_data_preview_controller = LabelDataPreviewController(self)
        # 数据已经是yolo,hrnet格式，可视化训练集和验证集
        self.train_data_visualization_controller = TrainDataVisualizationController(self)
        # 利用数据，开始训练
        self.train_controller = TrainController(self)
        # 利用训练好的模型，进行推理，并且可视化结果
        self.inference_controller = InferenceController(self)

        self._configure_work_directory_selector()
        self._configure_folder_tree_view()
        self.label_data_preview_controller.configure_ui()
        self.inference_controller.configure_ui()
        self.train_data_visualization_controller.configure_view()
        self._connect_signals()

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        if not self._restore_persisted_state():
            self.label_data_preview_controller.refresh_preview_label_scan_result([])

        self.setWindowState(self.windowState() | Qt.WindowMaximized)

    def append_log(self, message: str):
        print(message, flush=True)
        if threading.current_thread() is threading.main_thread():
            self._append_log_message_ui(message)
        else:
            self.log_message.emit(message)

    def _append_log_message_ui(self, message: str):
        if hasattr(self, "logTextBrowser") and self.logTextBrowser is not None:
            self.logTextBrowser.append(message)

    def get_work_dir_path(self):
        return Workspace.from_text(self.comboBoxWorkDirectory.currentText().strip())

    def get_workspace(self):
        return self.get_work_dir_path()

    def mark_image_list_dirty(self):
        self.is_image_list_dirty = True
        self.current_index = -1

    def keyPressEvent(self, event):
        if self.train_data_visualization_controller.handle_key_press(
            event.key(), event.modifiers(), event.isAutoRepeat()
        ):
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        try:
            self._save_current_work_directory_state()
        finally:
            super().closeEvent(event)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.KeyPress:
            if self.train_data_visualization_controller.handle_key_press(
                event.key(), event.modifiers(), event.isAutoRepeat()
            ):
                return True
        return super().eventFilter(watched, event)

    def _configure_work_directory_selector(self):
        self.comboBoxWorkDirectory.setEditable(True)
        self.comboBoxWorkDirectory.setStyleSheet(work_directory_combo_stylesheet())
        self.comboBoxWorkDirectory.clear()
        self.comboBoxWorkDirectory.addItems(self.recent_work_directories)
        self.comboBoxWorkDirectory.setCurrentIndex(-1)
        self.comboBoxWorkDirectory.clearEditText()

    def _configure_folder_tree_view(self):
        self._folder_tree_model.setRootPath("")
        self.folderTreeView.setModel(self._folder_tree_model)
        self.folderTreeView.setRootIndex(QModelIndex())
        for column in range(1, self._folder_tree_model.columnCount()):
            self.folderTreeView.hideColumn(column)

        selection_model = self.folderTreeView.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(self._on_folder_tree_selection_changed)
        else:
            def _connect_selection_model():
                sel_model = self.folderTreeView.selectionModel()
                if sel_model is not None:
                    sel_model.selectionChanged.connect(self._on_folder_tree_selection_changed)

            QTimer.singleShot(0, _connect_selection_model)

        current_directory = self.comboBoxWorkDirectory.currentText().strip()
        if current_directory:
            self._update_folder_tree_view(current_directory)

    def _on_work_directory_text_changed(self, directory):
        self._update_folder_tree_view(directory)
        self._restore_persisted_state(directory)

    def _on_folder_tree_selection_changed(self, selected, deselected):
        self.train_data_visualization_controller.on_folder_tree_changed(selected, deselected)
        self.inference_controller.on_folder_tree_changed(selected, deselected)

    def _on_model_task_changed(self, checked=False):
        if not checked:
            return
        self.train_controller.sync_task_ui()
        self.inference_controller.sync_task_ui()


    def _on_main_tab_changed(self, index):
        tab_widget = getattr(self, "tabWidgetMain", None)
        if tab_widget is None:
            return

        current_widget = tab_widget.widget(index)
        if current_widget is self.tabPreview:
            self.label_data_preview_controller.on_enter()
        elif current_widget is self.tabVisualTrain:
            self.train_data_visualization_controller.on_enter()
        elif current_widget is self.tabTrain:
            self.train_controller.on_enter()
        elif current_widget is self.tabInference:
            self.inference_controller.on_enter()

    def _connect_signals(self):
        self.btnSelectWorkDirectory.clicked.connect(self.choose_work_directory)
        self.comboBoxWorkDirectory.currentTextChanged.connect(self._on_work_directory_text_changed)
        self.btnDataProcessing.clicked.connect(self.data_prepare_controller.run_data_processing)
        self.btnScanJson.clicked.connect(self.label_data_preview_controller.scan_group_data_labels)
        self.btnSplitTrainData.clicked.connect(self.label_data_preview_controller.run_train_data_split)
        self.btnLastImage.clicked.connect(self.train_data_visualization_controller.show_previous_image)
        self.btnNextImage.clicked.connect(self.train_data_visualization_controller.show_next_image)
        self.tabWidgetMain.currentChanged.connect(self._on_main_tab_changed)

        for radio_name in ("radioTaskKeypoint", "radioTaskObb"):
            radio = getattr(self, radio_name, None)
            if radio is not None:
                radio.toggled.connect(self._on_model_task_changed)

        # YOLO 模型大小单选按钮：切换时同步预训练权重文本框
        for size in ("n", "s", "m", "l", "x"):
            radio = getattr(self, f"radioYoloSize{size.upper()}", None)
            if radio is not None:
                radio.toggled.connect(self._on_model_task_changed)

        self.btnTrainYolo.clicked.connect(self.train_controller.run_yolo_train)
        self.btnTrainHrnet.clicked.connect(self.train_controller.run_hrnet_train)
        self.train_controller.sync_task_ui()
        self.inference_controller.sync_task_ui()
        self.btnExportYoloOnnx.clicked.connect(self.inference_controller.export_yolo_onnx)
        self.btnExportHrnetOnnx.clicked.connect(self.inference_controller.export_hrnet_onnx)
        self.btnExportYoloEngine.clicked.connect(
            lambda: self.inference_controller.export_engine("yolo")
        )
        self.btnExportEngine.clicked.connect(
            lambda: self.inference_controller.export_engine("hrnet")
        )
        self.btnSelectTensorrtPath.clicked.connect(self.inference_controller.select_tensorrt_path)
        self.btnBatchInfer.clicked.connect(self.inference_controller.run_batch_inference)
        self.checkBoxSaveJson.toggled.connect(self.inference_controller.on_save_json_toggled)
        self.checkBoxSavePartJson.toggled.connect(self.inference_controller.on_save_part_json_toggled)

    def _load_recent_work_directories(self):
        if CONFIG_FILE.exists():
            try:
                with CONFIG_FILE.open("r", encoding="utf-8") as file:
                    payload = json.load(file)
            except (OSError, json.JSONDecodeError):
                payload = {}

            if isinstance(payload, list):
                stored_directories = payload
            elif isinstance(payload, dict):
                stored_directories = payload.get(RECENT_WORK_DIRECTORY_KEY, [])
            else:
                stored_directories = []
        else:
            stored_directories = []

        if isinstance(stored_directories, str):
            stored_directories = [stored_directories]

        recent_directories = []
        for directory in stored_directories or []:
            normalized_directory = str(Path(str(directory)).expanduser())
            if normalized_directory and normalized_directory not in recent_directories:
                recent_directories.append(normalized_directory)

        return recent_directories[:MAX_RECENT_WORK_DIRECTORIES]

    def _save_recent_work_directories(self):
        payload = {RECENT_WORK_DIRECTORY_KEY: self.recent_work_directories}
        try:
            with CONFIG_FILE.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _as_work_dir_path(self, work_dir_path=None):
        if work_dir_path is None:
            workspace = self.get_workspace()
            return None if workspace is None else workspace.root
        if isinstance(work_dir_path, Workspace):
            return work_dir_path.root
        return Path(work_dir_path).expanduser()

    def _get_work_directory_config_path(self, work_dir_path=None):
        resolved_path = self._as_work_dir_path(work_dir_path)
        if resolved_path is None:
            return None
        return resolved_path / WORK_DIRECTORY_CONFIG_NAME

    @staticmethod
    def _read_json_payload(file_path: Path):
        try:
            with file_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            return None
        if isinstance(payload, dict):
            return payload
        return None

    @staticmethod
    def _write_json_payload(file_path: Path, payload: dict):
        try:
            with file_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _collect_widget_state(self):
        widget_state = {}
        for widget in self.findChildren(QWidget):
            widget_name = widget.objectName()
            if not widget_name or widget_name.startswith("qt_"):
                continue

            try:
                if isinstance(widget, QLineEdit):
                    widget_state[widget_name] = {"kind": "QLineEdit", "text": widget.text()}
                elif isinstance(widget, QComboBox):
                    try:
                        current_data = widget.currentData()
                    except Exception:
                        current_data = None
                    if not isinstance(current_data, (str, int, float, bool)) and current_data is not None:
                        current_data = str(current_data)
                    widget_state[widget_name] = {
                        "kind": "QComboBox",
                        "text": widget.currentText(),
                        "data": current_data,
                        "index": widget.currentIndex(),
                    }
                elif isinstance(widget, QSpinBox):
                    widget_state[widget_name] = {"kind": "QSpinBox", "value": int(widget.value())}
                elif isinstance(widget, QDoubleSpinBox):
                    widget_state[widget_name] = {
                        "kind": "QDoubleSpinBox",
                        "value": float(widget.value()),
                    }
                elif isinstance(widget, QCheckBox):
                    widget_state[widget_name] = {"kind": "QCheckBox", "checked": bool(widget.isChecked())}
                elif isinstance(widget, QRadioButton):
                    widget_state[widget_name] = {
                        "kind": "QRadioButton",
                        "checked": bool(widget.isChecked()),
                    }
            except Exception:
                continue

        tab_widget = getattr(self, "tabWidgetMain", None)
        if tab_widget is not None:
            try:
                widget_state["__tabWidgetMain__"] = int(tab_widget.currentIndex())
            except Exception:
                pass

        checked_labels = self.inference_controller.collect_save_part_label_checked()
        if checked_labels:
            widget_state["__save_part_label_checked__"] = checked_labels

        return widget_state

    def _restore_combo_box_state(self, combo_box, state):
        if combo_box is None or not isinstance(state, dict):
            return

        preferred_data = state.get("data")
        preferred_text = state.get("text", "")
        preferred_index = state.get("index")

        try:
            if preferred_data not in (None, ""):
                found_index = combo_box.findData(preferred_data)
                if found_index >= 0:
                    combo_box.setCurrentIndex(found_index)
                    return
        except Exception:
            pass

        try:
            if preferred_text:
                found_index = combo_box.findText(str(preferred_text))
                if found_index >= 0:
                    combo_box.setCurrentIndex(found_index)
                    return
                if combo_box.isEditable():
                    combo_box.setEditText(str(preferred_text))
                    return
        except Exception:
            pass

        try:
            if preferred_index is not None:
                combo_box.setCurrentIndex(int(preferred_index))
        except Exception:
            pass

    def _restore_widget_state(self, widget_state):
        if not isinstance(widget_state, dict):
            return

        for widget in self.findChildren(QWidget):
            widget_name = widget.objectName()
            if not widget_name or widget_name.startswith("qt_"):
                continue

            state = widget_state.get(widget_name)
            if not isinstance(state, dict):
                continue

            try:
                if isinstance(widget, QLineEdit):
                    widget.blockSignals(True)
                    widget.setText(str(state.get("text", "")))
                    widget.blockSignals(False)
                elif isinstance(widget, QComboBox):
                    widget.blockSignals(True)
                    self._restore_combo_box_state(widget, state)
                    widget.blockSignals(False)
                elif isinstance(widget, QSpinBox):
                    widget.blockSignals(True)
                    widget.setValue(int(state.get("value", widget.value())))
                    widget.blockSignals(False)
                elif isinstance(widget, QDoubleSpinBox):
                    widget.blockSignals(True)
                    widget.setValue(float(state.get("value", widget.value())))
                    widget.blockSignals(False)
                elif isinstance(widget, QCheckBox):
                    widget.blockSignals(True)
                    widget.setChecked(bool(state.get("checked", False)))
                    widget.blockSignals(False)
                elif isinstance(widget, QRadioButton):
                    # Only force-check the saved selection; exclusive groups
                    # will auto-uncheck the other radios.
                    if bool(state.get("checked", False)):
                        widget.blockSignals(True)
                        widget.setChecked(True)
                        widget.blockSignals(False)
            except Exception:
                continue

        tab_widget = getattr(self, "tabWidgetMain", None)
        if tab_widget is not None:
            try:
                tab_widget.setCurrentIndex(
                    int(widget_state.get("__tabWidgetMain__", tab_widget.currentIndex()))
                )
            except Exception:
                pass

    def _build_persisted_payload(self):
        return {
            "version": CONFIG_VERSION,
            "widgetState": self._collect_widget_state(),
            "labelMappingRows": self.label_data_preview_controller.collect_label_mapping_rows(),
        }

    def _save_current_work_directory_state(self):
        work_dir_path = self._as_work_dir_path()
        if work_dir_path is None:
            self._save_recent_work_directories()
            return

        normalized_directory = str(work_dir_path)
        if normalized_directory in self.recent_work_directories:
            self.recent_work_directories.remove(normalized_directory)
        self.recent_work_directories.insert(0, normalized_directory)
        self.recent_work_directories = self.recent_work_directories[:MAX_RECENT_WORK_DIRECTORIES]
        self._save_recent_work_directories()

        config_path = self._get_work_directory_config_path(work_dir_path)
        if config_path is None:
            return

        payload = self._build_persisted_payload()
        payload["workDirectory"] = normalized_directory
        self._write_json_payload(config_path, payload)

    def _restore_persisted_state(self, work_directory=None):
        if work_directory is None:
            if not self.recent_work_directories:
                return False

            for candidate in self.recent_work_directories:
                candidate_path = Path(candidate).expanduser()
                if candidate_path.is_dir():
                    work_directory = str(candidate_path)
                    break

            if work_directory is None:
                work_directory = str(Path(self.recent_work_directories[0]).expanduser())
        else:
            work_directory = str(Path(work_directory).expanduser())

        self.comboBoxWorkDirectory.blockSignals(True)
        if self.comboBoxWorkDirectory.findText(work_directory) < 0:
            self.comboBoxWorkDirectory.addItem(work_directory)
        self.comboBoxWorkDirectory.setCurrentText(work_directory)
        self.comboBoxWorkDirectory.blockSignals(False)

        self._update_folder_tree_view(work_directory)
        self.inference_controller.refresh_model_lists()

        config_path = self._get_work_directory_config_path(work_directory)
        if config_path is None or not config_path.exists():
            return False

        payload = self._read_json_payload(config_path)
        if payload is None:
            return False

        widget_state = payload.get("widgetState", {})
        if not isinstance(widget_state, dict):
            widget_state = {}
        self.saved_widget_state = widget_state

        label_rows = payload.get("labelMappingRows", [])
        self.label_data_preview_controller.restore_label_mapping_rows(label_rows)
        self._restore_widget_state(self.saved_widget_state)
        self.inference_controller.restore_save_part_label_panel(self.saved_widget_state)
        return True

    def _set_work_directory(self, directory: str):
        normalized_directory = str(Path(directory).expanduser())
        if normalized_directory in self.recent_work_directories:
            self.recent_work_directories.remove(normalized_directory)

        self.recent_work_directories.insert(0, normalized_directory)
        self.recent_work_directories = self.recent_work_directories[:MAX_RECENT_WORK_DIRECTORIES]
        self._save_recent_work_directories()

        self.comboBoxWorkDirectory.blockSignals(True)
        self.comboBoxWorkDirectory.clear()
        self.comboBoxWorkDirectory.addItems(self.recent_work_directories)
        self.comboBoxWorkDirectory.setCurrentText(normalized_directory)
        self.comboBoxWorkDirectory.blockSignals(False)
        self._update_folder_tree_view(normalized_directory)
        self.mark_image_list_dirty()

        # 恢复该目录之前保存的控件状态
        self._restore_persisted_state(normalized_directory)

    def _update_folder_tree_view(self, directory: str):
        normalized_directory = str(Path(directory).expanduser()).strip()
        directory_path = Path(normalized_directory)
        self.is_image_list_dirty = True

        if normalized_directory and directory_path.is_dir():
            model_index = self._folder_tree_model.setRootPath(str(directory_path))
            self.folderTreeView.setRootIndex(model_index)
            return

        self.folderTreeView.setRootIndex(QModelIndex())

    def choose_work_directory(self, target_line_edit=None):
        start_directory = self.comboBoxWorkDirectory.currentText().strip()
        if not start_directory:
            start_directory = str(Path.home())

        selected_directory = QFileDialog.getExistingDirectory(self, "选择文件夹", start_directory)
        if not selected_directory:
            return
        self._set_work_directory(selected_directory) 

    def get_selected_folder_tree_path(self):
        tree_index = self.folderTreeView.currentIndex()
        if not tree_index.isValid():
            return ""

        tree_model = self.folderTreeView.model()
        if tree_model is None or not hasattr(tree_model, "filePath"):
            return ""

        selected_path = str(Path(tree_model.filePath(tree_index)).expanduser()).strip()
        if not selected_path or not Path(selected_path).is_dir():
            return ""
        return selected_path
