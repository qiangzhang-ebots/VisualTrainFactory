import multiprocessing
from pathlib import Path
import contextlib
import io
import json
import sys
import threading
import time
import shutil

from ImageView import ImageView
from s0_dataprocessing import find_images, process_data
from s1_dataJson2Train import ConvertInfo, process_filesHRNet, process_filesYolo
from s2_visualTrainData import visual_Yolo_trainData


def _import_qt_widgets():
    try:
        from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QFileSystemModel, QLabel, QLineEdit, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QComboBox, QHeaderView, QAbstractItemView
        from PySide6.QtCore import QModelIndex, QEvent, QObject, Qt

        return QApplication, QMainWindow, QFileDialog, QFileSystemModel, QLabel, QLineEdit, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QComboBox, QHeaderView, QAbstractItemView, QModelIndex, QEvent, QObject, Qt, "pyside6"
    except ImportError:
        pass

    try:
        from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QFileSystemModel, QLabel, QLineEdit, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QComboBox, QHeaderView, QAbstractItemView
        from PyQt5.QtCore import QModelIndex, QEvent, QObject, Qt

        return QApplication, QMainWindow, QFileDialog, QFileSystemModel, QLabel, QLineEdit, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QComboBox, QHeaderView, QAbstractItemView, QModelIndex, QEvent, QObject, Qt, "pyqt5"
    except ImportError:
        pass

    try:
        from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QFileSystemModel, QLabel, QLineEdit, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QComboBox, QHeaderView, QAbstractItemView
        from PySide2.QtCore import QModelIndex, QEvent, QObject, Qt

        return QApplication, QMainWindow, QFileDialog, QFileSystemModel, QLabel, QLineEdit, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QComboBox, QHeaderView, QAbstractItemView, QModelIndex, QEvent, QObject, Qt, "pyside2"
    except ImportError as exc:
        raise ImportError("Need PySide6, PyQt5, or PySide2 installed") from exc


def _load_ui_into(window, ui_path: Path, backend: str):
    if backend == "pyqt5":
        from PyQt5 import uic

        uic.loadUi(str(ui_path), window)
        return

    if backend == "pyside6":
        from PySide6 import QtUiTools
        from PySide6.QtCore import QFile, QIODevice

    else:
        from PySide2 import QtUiTools
        from PySide2.QtCore import QFile, QIODevice

    file = QFile(str(ui_path))
    if not file.open(QIODevice.ReadOnly):
        raise FileNotFoundError(f"Cannot open UI file: {ui_path}")

    loader = QtUiTools.QUiLoader()
    loaded = loader.load(file, window)
    file.close()

    if loaded is None:
        raise RuntimeError(f"Failed to load UI file: {ui_path}")

    window.setCentralWidget(loaded.centralWidget())
    window.setMenuBar(loaded.menuBar())
    window.setStatusBar(loaded.statusBar())
    window.resize(loaded.size())
    window.setWindowTitle(loaded.windowTitle())


QApplication, QMainWindow, QFileDialog, QFileSystemModel, QLabel, QLineEdit, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QComboBox, QHeaderView, QAbstractItemView, QModelIndex, QEvent, QObject, Qt, QT_BACKEND = _import_qt_widgets()


RECENT_WORK_DIRECTORY_KEY = "recentWorkDirectories"
MAX_RECENT_WORK_DIRECTORIES = 10
CONFIG_FILE = Path(__file__).with_name("VisualFactoryConfig.json")
COMBO_ARROW_ICON = Path(__file__).with_name("combo-arrow-down.svg")
COMBO_ARROW_LIGHT_ICON = Path(__file__).with_name("combo-arrow-down-light.svg")
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


class VisualTrainFactoryWindow(QMainWindow):
    def __init__(self, parent=None):
        """初始化主窗口并完成 UI、数据和信号的基础装配。"""
        super().__init__(parent)
        ui_path = Path(__file__).with_name("VisualTrainFactory.ui")
        _load_ui_into(self, ui_path, QT_BACKEND)
        self._folder_tree_model = QFileSystemModel(self)
        self._recent_work_directories = self._load_recent_work_directories()
        self._label_id_edits = {}
        self._label_usage_combos = {}
        self._visual_train_image_pairs = []
        self._visual_train_current_index = -1
        # 标记训练数据可视化页的图片/标签列表是否需要刷新。
        # 只要工作目录、数据划分等发生变化，设置为 True，下一次切换到可视化页或翻页时会重新扫描 datasets 目录。
        # 这样可以避免图片/标签列表因外部变动而过时，保证翻页和显示始终和最新数据同步。
        self._visual_train_dirty = True
        self._configure_work_directory_selector()
        self._configure_folder_tree_view()
        self._configure_label_mapping_area()
        self._configure_visual_train_view()
        self._connect_signals()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._refresh_preview_label_scan_result([])

    def keyPressEvent(self, event):
        """在训练数据可视化页支持 A/D 快捷键翻页。"""
        if self._handle_visual_train_key_press(event.key(), event.modifiers(), event.isAutoRepeat()):
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched, event):
        """拦截全局按键事件，让训练可视化页不依赖焦点也能接收 A/D。"""
        if event.type() == QEvent.KeyPress:
            if self._handle_visual_train_key_press(event.key(), event.modifiers(), event.isAutoRepeat()):
                return True
        return super().eventFilter(watched, event)

    def _handle_visual_train_key_press(self, key, modifiers=None, is_auto_repeat=False):
        """当训练数据可视化页激活时，统一处理 A/D 翻页。"""
        tab_widget = getattr(self, 'tabWidgetMain', None)
        if tab_widget is None or not hasattr(self, 'tabVisualTrain'):
            return False

        current_widget = tab_widget.currentWidget()
        if current_widget is not self.tabVisualTrain:
            return False

        if modifiers not in (None, Qt.NoModifier):
            return False

        if is_auto_repeat:
            return False

        if key == Qt.Key_A:
            self.visual_train_last_image_slot()
            return True

        if key == Qt.Key_D:
            self.visual_train_next_image_slot()
            return True

        return False
        
    def _configure_work_directory_selector(self):
        """把工作目录输入框配置成可编辑下拉框，并恢复历史目录。"""
        self.workDirectoryLineEdit.setEditable(True)
        arrow_icon_path = COMBO_ARROW_ICON.resolve().as_posix()
        self.workDirectoryLineEdit.setStyleSheet(
            f"""
            QComboBox {{
                padding-right: 34px;
            }}

            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 28px;
                border: none;
            }}

            QComboBox::down-arrow {{
                image: url({arrow_icon_path});
                width: 12px;
                height: 8px;
            }}
            """
        )
        self.workDirectoryLineEdit.clear()
        self.workDirectoryLineEdit.addItems(self._recent_work_directories)
        self.workDirectoryLineEdit.setCurrentIndex(-1)
        self.workDirectoryLineEdit.clearEditText()

    def _configure_folder_tree_view(self):
        """初始化左侧文件树，并在已有工作目录时同步到对应路径。"""
        self._folder_tree_model.setRootPath("")
        self.folderTreeView.setModel(self._folder_tree_model)
        self.folderTreeView.setRootIndex(QModelIndex())
        for column in range(1, self._folder_tree_model.columnCount()):
            self.folderTreeView.hideColumn(column)

        # 连接 selectionChanged 信号，确保切换文件夹时刷新图片列表
        selection_model = self.folderTreeView.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(self._on_folder_tree_selection_changed)
        else:
            # 兼容初始化时 selectionModel 可能为 None 的情况，延迟连接
            def _connect_selection_model():
                sel_model = self.folderTreeView.selectionModel()
                if sel_model is not None:
                    sel_model.selectionChanged.connect(self._on_folder_tree_selection_changed)
            QTimer = None
            try:
                from PySide6.QtCore import QTimer
            except ImportError:
                try:
                    from PyQt5.QtCore import QTimer
                except ImportError:
                    from PySide2.QtCore import QTimer
            QTimer.singleShot(0, _connect_selection_model)

        current_directory = self.workDirectoryLineEdit.currentText().strip()
        if current_directory:
            self._update_folder_tree_view(current_directory)

    def _on_folder_tree_selection_changed(self, selected, deselected):
        """当文件树选择变化时，刷新可视化图片列表。"""
        self._visual_train_dirty = True
        self._visual_train_current_index = -1
        self._refresh_visual_train_image_list(force=True)
        self._show_visual_train_current_image()
        # 当用户在文件树中选择文件夹时，实时把所选文件夹路径写入推理页的输入框（如果存在）
        try:
            selected_path = self._get_selected_folder_tree_path()
            if selected_path and hasattr(self, 'inferImgForlderLineEdit') and self.inferImgForlderLineEdit is not None:
                self.inferImgForlderLineEdit.blockSignals(True)
                self.inferImgForlderLineEdit.setText(selected_path)
                self.inferImgForlderLineEdit.blockSignals(False)
        except Exception:
            pass

    def _load_recent_work_directories(self):
        """从配置文件读取最近使用过的工作目录列表。"""
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
        """把最近使用的工作目录写回配置文件。"""
        payload = {RECENT_WORK_DIRECTORY_KEY: self._recent_work_directories}

        try:
            with CONFIG_FILE.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _set_work_directory(self, directory: str):
        """切换当前工作目录，并刷新历史记录和文件树。"""
        normalized_directory = str(Path(directory).expanduser())

        if normalized_directory in self._recent_work_directories:
            self._recent_work_directories.remove(normalized_directory)

        self._recent_work_directories.insert(0, normalized_directory)
        self._recent_work_directories = self._recent_work_directories[:MAX_RECENT_WORK_DIRECTORIES]
        self._save_recent_work_directories()

        self.workDirectoryLineEdit.blockSignals(True)
        self.workDirectoryLineEdit.clear()
        self.workDirectoryLineEdit.addItems(self._recent_work_directories)
        self.workDirectoryLineEdit.setCurrentText(normalized_directory)
        self.workDirectoryLineEdit.blockSignals(False)
        self._update_folder_tree_view(normalized_directory)
        self._visual_train_dirty = True
        self._visual_train_current_index = -1

    def _update_folder_tree_view(self, directory: str):
        """根据输入的工作目录同步文件树根节点。"""
        normalized_directory = str(Path(directory).expanduser()).strip()
        directory_path = Path(normalized_directory)

        self._visual_train_dirty = True

        if normalized_directory and directory_path.is_dir():
            model_index = self._folder_tree_model.setRootPath(str(directory_path))
            self.folderTreeView.setRootIndex(model_index)
            return

        self.folderTreeView.setRootIndex(QModelIndex())

    def _connect_signals(self):
        """把按钮、下拉框和标签页切换事件连接到对应处理函数。"""
        self.selectWorkDirectoryBtn.clicked.connect(self.choose_working_directory_slot)
        self.workDirectoryLineEdit.currentTextChanged.connect(self._update_folder_tree_view)
        self.dataProcessingBtn.clicked.connect(self.data_processing_slot)
        self.scanJsonBtn.clicked.connect(self.scan_group_data_labels_slot)
        self.splitTrainDataBtn.clicked.connect(self.split_TrainData_slot)
        self.lastImgBtn.clicked.connect(self.visual_train_last_image_slot)
        self.nextImgBtn.clicked.connect(self.visual_train_next_image_slot)
        self.tabWidgetMain.currentChanged.connect(self._handle_main_tab_changed)

        self.YoloTrainBtn.clicked.connect(self.yolo_train_slot)
        self.HRNetTrainBtn.clicked.connect(self.hrnet_train_slot) 
        self.exportYoloOnnxBtn.clicked.connect(self.export_yolo_onnx_slot)
        self.exportHRNetOnnxBtn.clicked.connect(self.export_hrnet_onnx_slot)
        # 批量推理按钮
        self.batchInferBtn.clicked.connect(self.batch_infer_slot)

    def _get_selected_model_path_from_combo(self, combo_name: str):
        combo = getattr(self, combo_name, None)
        if combo is None:
            return None

        try:
            model_path = combo.currentData()
        except Exception:
            model_path = None

        if isinstance(model_path, str):
            model_path = model_path.strip()

        return model_path or None

    def export_yolo_onnx_slot(self):
        """导出当前选择的 YOLO 训练结果对应的 ONNX 模型。"""
        import traceback

        try:
            from s5_exportOnnx import exportYoloOnnx

            work_dir = self._get_work_directory_path()
            if work_dir is None:
                self._append_log_message('请先选择有效的工作目录。')
                return

            yolo_model_path = self._get_selected_model_path_from_combo('YoloModelCombbox')
            if not yolo_model_path:
                self._append_log_message('请先在推理页选择一个 YOLO 模型。')
                return

            self._append_log_message(f'开始导出 YOLO ONNX: {yolo_model_path}')
            exportYoloOnnx(yolo_model_path)
            self._append_log_message(f'YOLO ONNX 导出完成')
        except Exception as exc:
            self._append_log_message(f'YOLO ONNX 导出失败: {exc}\n{traceback.format_exc()}')

    def export_hrnet_onnx_slot(self):
        """导出当前选择的 HRNet 训练结果对应的 ONNX 模型。"""
        import traceback

        try:
            from s5_exportOnnx import exportHRNetOnnx

            work_dir = self._get_work_directory_path()
            if work_dir is None:
                self._append_log_message('请先选择有效的工作目录。')
                return

            hrnet_model_path = self._get_selected_model_path_from_combo('HRNetModelCombbox')
            if not hrnet_model_path:
                self._append_log_message('请先在推理页选择一个 HRNet 模型。')
                return

            self._append_log_message(f'开始导出 HRNet ONNX: {hrnet_model_path}')
            onnx_path = exportHRNetOnnx(hrnet_model_path)
            self._append_log_message(f'HRNet ONNX 导出完成: {onnx_path}')
        except Exception as exc:
            self._append_log_message(f'HRNet ONNX 导出失败: {exc}\n{traceback.format_exc()}')

    def yolo_train_slot(self):
        """YOLO训练按钮的槽函数，调用s3_train.py的trainYolo。"""
        import traceback
        try:
            from s3_train import trainYolo, _get_log_name
            work_dir = self.workDirectoryLineEdit.currentText().strip()
            if not work_dir:
                self._append_log_message("请先选择工作目录！")
                return
            label_map = self.get_label_id_mapping()
            if not label_map:
                self._append_log_message("请先扫描并填写标签ID映射！")
                return
            # 训练参数从界面控件读取
            epochs = 100
            img_size = 640
            batch_size = 16
            gpu = '0'
            hflipRatio = 0.0
            vflipRatio = 0.0
            works = 8
            # 读取hflipRatio
            if hasattr(self, 'hflipLineEdit'):
                try:
                    hflipRatio = float(self.hflipLineEdit.text())
                except Exception:
                    pass
            # 读取vflipRatio
            if hasattr(self, 'vflipLineEdit'):
                try:
                    vflipRatio = float(self.vflipLineEdit.text())
                except Exception:
                    pass
            # 读取works
            if hasattr(self, 'spinBoxYoloWorkers'):
                try:
                    works = int(self.spinBoxYoloWorkers.value())
                except Exception:
                    pass
            # 读取epochs
            if hasattr(self, 'spinBoxYoloEpochs'):
                try:
                    epochs = int(self.spinBoxYoloEpochs.value())
                except Exception:
                    pass
            # 读取img_size
            if hasattr(self, 'spinBoxYoloImgSize'):
                try:
                    img_size = int(self.spinBoxYoloImgSize.value())
                except Exception:
                    pass
            # 读取batch_size（如有控件可补充）
            if hasattr(self, 'spinBoxYoloBatch'):
                try:
                    batch_size = int(self.spinBoxYoloBatch.value())
                except Exception:
                    pass

            if hasattr(self, 'lineEditYoloDevice'):
                gpu_text = self.lineEditYoloDevice.text()
                if gpu_text:
                    gpu = gpu_text
            logName = _get_log_name()
            self._append_log_message("开始YOLO训练...")
            # 调用训练
            trainYolo(
                work_dir, label_map, epochs, batch_size, img_size,
                gpu=gpu, logName=logName,
                workers=works, hflipRatio=hflipRatio, vflipRatio=vflipRatio
            )
            self._append_log_message("YOLO训练已完成。")
        except Exception as e:
            self._append_log_message(f"YOLO训练启动失败: {e}\n{traceback.format_exc()}")

    def hrnet_train_slot(self):
        """HRNet 训练按钮的槽函数，调用 s3_train 中的 trainHRNet。

        s3_train.trainHRNet 的签名是:
            trainHRNet(workspace, epochs, batch_size, img_size, gpu, logName)
        因此不需要传入标签映射，直接按签名调用。
        """
        import traceback
        try:
            from s3_train import trainHRNet, _get_log_name
            work_dir = self.workDirectoryLineEdit.currentText().strip()
            if not work_dir:
                self._append_log_message("请先选择工作目录！")
                return

            # 默认训练参数，可由界面控件覆盖
            epochs = 100
            img_size = 640
            batch_size = 16
            gpu = '0'

            if hasattr(self, 'spinBoxHrnetEpochs'):
                try:
                    epochs = int(self.spinBoxHrnetEpochs.value())
                except Exception:
                    pass

            if hasattr(self, 'HRNetImgSizeLineEdit'):
                try:
                    img_size = int(self.HRNetImgSizeLineEdit.text())
                except Exception:
                    pass

            if hasattr(self, 'spinBoxHrnetBatch'):
                try:
                    batch_size = int(self.spinBoxHrnetBatch.value())
                except Exception:
                    pass

            if hasattr(self, 'lineEditHrnetGpu'):
                gpu_text = self.lineEditHrnetGpu.text()
                if gpu_text:
                    gpu = gpu_text

            gpu_ids = [item.strip() for item in str(gpu).split(',') if item.strip()]
            if len(gpu_ids) > 1:
                per_gpu_batch = max(1, (batch_size + len(gpu_ids) - 1) // len(gpu_ids))
                self._append_log_message(
                    'HRNet 将以多卡分布式模式启动: '
                    f'devices={gpu_ids}, 总 batch={batch_size}, 每卡 batch={per_gpu_batch}'
                )
            else:
                self._append_log_message(f'HRNet 将以单卡模式启动: device={gpu}, batch={batch_size}')

            logName = _get_log_name()
            self._append_log_message("开始HRNet训练...")

            # 按 s3_train.trainHRNet 的签名调用
            trainHRNet(
                work_dir,
                epochs,
                batch_size,
                img_size,
                gpu,
                logName,
            )

            self._append_log_message("HRNet训练已完成。")
        except Exception as e:
            self._append_log_message(f"HRNet训练启动失败: {e}\n{traceback.format_exc()}")

    def _configure_visual_train_view(self):
        """为训练数据可视化页插入 ImageView，并准备翻页状态。"""
        if not hasattr(self, 'verticalLayout_2'):
            self._visual_train_image_view = None
            return

        self._visual_train_image_view = ImageView(self.tabVisualTrain if hasattr(self, 'tabVisualTrain') else self)
        self._visual_train_image_view.setObjectName('visualTrainImageView')
        self._visual_train_image_view.setMinimumHeight(420)
        self._visual_train_image_view.setStyleSheet('background-color: #000000; border: 1px solid #334155;')
        self.verticalLayout_2.addWidget(self._visual_train_image_view, 1)

        self._refresh_visual_train_image_list(force=True)
        self._show_visual_train_image(0)

    def _append_log_message(self, message: str):
        if hasattr(self, 'logTextBrowser') and self.logTextBrowser is not None:
            self.logTextBrowser.append(message)
        print(message)

    def _configure_label_mapping_area(self):
        """构建标签映射表格，用于扫描后展示 label、类型和训练 id。"""
        if not hasattr(self, 'labelMapContainer') or self.labelMapContainer is None:
            self.labelTable = None
            return

        self.labelMapContainer.setMinimumHeight(220)

        self.labelTable = QTableWidget(self.labelMapContainer)
        self.labelTable.setColumnCount(4)
        self.labelTable.setHorizontalHeaderLabels(['Label', '类型', '训练时ID', '请选择用途'])
        self.labelTable.setAlternatingRowColors(True)
        self.labelTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.labelTable.verticalHeader().setVisible(False)
        self.labelTable.setStyleSheet(
            "QTableWidget {"
            "  background-color: #0f172a;"
            "  color: #f8fafc;"
            "  gridline-color: #334155;"
            "  alternate-background-color: #111827;"
            "}"
            "QHeaderView::section {"
            "  background-color: #1e293b;"
            "  color: #f8fafc;"
            "  border: 1px solid #334155;"
            "  padding: 4px;"
            "}"
            "QTableWidget::item:selected {"
            "  background: #2563eb;"
            "  color: #ffffff;"
            "}"
        )
        self.labelTable.horizontalHeader().setStretchLastSection(False)
        self.labelTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.labelTable.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.labelTable.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.labelTable.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.labelTable.setMinimumHeight(220)

        outer_layout = QVBoxLayout(self.labelMapContainer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.labelTable)

        self._populate_label_mapping_layout([])

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _populate_label_mapping_layout(self, labels, label_stats=None):
        """根据扫描结果动态生成标签映射行，并保留已有输入。"""
        if not hasattr(self, 'labelTable') or self.labelTable is None:
            return

        label_stats = label_stats or {}
        arrow_icon_path = COMBO_ARROW_LIGHT_ICON.resolve().as_posix()

        previous_values = {}
        for label_text, line_edit in self._label_id_edits.items():
            if line_edit is not None:
                previous_values[label_text] = line_edit.text().strip()

        previous_usages = {}
        for label_text, combo_box in self._label_usage_combos.items():
            if combo_box is not None:
                previous_usages[label_text] = combo_box.currentText()

        self._label_id_edits = {}
        self._label_usage_combos = {}
        self.labelTable.clearContents()
        self.labelTable.clearSpans()

        if not labels:
            self.labelTable.setRowCount(1)
            hint_item = QTableWidgetItem('扫描后将在这里显示每个 label 的类型和 id 映射。')
            self.labelTable.setItem(0, 0, hint_item)
            self.labelTable.setSpan(0, 0, 1, 4)
            return

        self.labelTable.setRowCount(len(labels))

        for row_index, label_text in enumerate(labels):
            type_text = self._format_label_type_text(label_stats.get(label_text, {}))

            label_item = QTableWidgetItem(label_text)
            type_item = QTableWidgetItem(type_text)

            self.labelTable.setItem(row_index, 0, label_item)
            self.labelTable.setItem(row_index, 1, type_item)

            row_input = QLineEdit(self.labelTable)
            row_input.setPlaceholderText('输入训练时对应的 id')
            row_input.setStyleSheet(
                "QLineEdit {"
                "  background-color: #0b1220;"
                "  color: #f8fafc;"
                "  border: 1px solid #475569;"
                "  border-radius: 4px;"
                "  padding: 2px 6px;"
                "}"
                "QLineEdit:focus {"
                "  border: 1px solid #60a5fa;"
                "}"
            )
            default_id_text = self._default_id_text_from_label(label_text)
            row_input.setText(previous_values.get(label_text, default_id_text))
            self.labelTable.setCellWidget(row_index, 2, row_input)

            usage_combo = QComboBox(self.labelTable)
            usage_options = ['请选择用途', '用于训练', '用于遮挡点']
            usage_combo.addItems(usage_options)
            default_usage = '用于训练' if default_id_text else '用于遮挡点'
            usage_combo.setCurrentText(previous_usages.get(label_text, default_usage))
            usage_combo.setStyleSheet(
                "QComboBox {"
                "  padding: 2px 28px 2px 8px;"
                "  border: 1px solid #475569;"
                "  border-radius: 4px;"
                "  background: #0b1220;"
                "  color: #f8fafc;"
                "}"
                "QComboBox::drop-down {"
                "  subcontrol-origin: padding;"
                "  subcontrol-position: top right;"
                "  width: 24px;"
                "  border-left: 1px solid #334155;"
                "  background: #1e293b;"
                "}"
                "QComboBox::drop-down:hover { background: #334155; }"
                "QComboBox QAbstractItemView {"
                "  background: #0b1220;"
                "  color: #f8fafc;"
                "  border: 1px solid #334155;"
                "  selection-background-color: #2563eb;"
                "  selection-color: #ffffff;"
                "}"
                f"QComboBox::down-arrow {{ image: url({arrow_icon_path}); width: 12px; height: 8px; }}"
            )
            self.labelTable.setCellWidget(row_index, 3, usage_combo)

            self._label_id_edits[label_text] = row_input
            self._label_usage_combos[label_text] = usage_combo

    def _default_id_text_from_label(self, label_text):
        try:
            return str(int(str(label_text).strip()))
        except (ValueError, TypeError):
            return ''

    def _format_label_type_text(self, stats):
        if not isinstance(stats, dict):
            return ''

        type_counter = stats.get('types') or {}
        if not type_counter:
            return ''

        primary_type = max(type_counter.items(), key=lambda item: (item[1], item[0]))[0]
        if primary_type == 'polygon':
            polygon_point_histogram = stats.get('polygon_point_histogram') or {}
            if isinstance(polygon_point_histogram, dict) and polygon_point_histogram:
                dominant_points, dominant_count = max(
                    polygon_point_histogram.items(),
                    key=lambda item: (item[1], item[0]),
                )
                if dominant_count > 0:
                    return f'polygon-{dominant_points}'

            polygon_max_points = int((stats.get('polygon_max_points') or 0))
            if polygon_max_points > 0:
                return f'polygon-{polygon_max_points}'
        return primary_type

    def _get_dominant_polygon_points(self, label_stats):
        if not isinstance(label_stats, dict) or not label_stats:
            return ''

        point_label_counts = {}
        for stats in label_stats.values():
            if not isinstance(stats, dict):
                continue

            polygon_point_histogram = stats.get('polygon_point_histogram') or {}
            if not isinstance(polygon_point_histogram, dict) or not polygon_point_histogram:
                continue

            dominant_points, _ = max(
                polygon_point_histogram.items(),
                key=lambda item: (item[1], item[0]),
            )
            point_label_counts[dominant_points] = point_label_counts.get(dominant_points, 0) + 1

        if not point_label_counts:
            return ''

        return str(max(point_label_counts.items(), key=lambda item: (item[1], item[0]))[0])

    def get_label_id_mapping(self):
        mapping = {}
        for label_text, line_edit in self._label_id_edits.items():
            if line_edit is None:
                continue
            value = line_edit.text().strip()
            if value:
                mapping[label_text] = int(value)
        return mapping

    def _get_work_directory_path(self):
        work_directory_text = self.workDirectoryLineEdit.currentText().strip()
        if not work_directory_text:
            return None

        work_directory_path = Path(work_directory_text).expanduser()
        if not work_directory_path.exists() or not work_directory_path.is_dir():
            return None

        return work_directory_path

    def _get_group_data_directory(self):
        work_directory_path = self._get_work_directory_path()
        if work_directory_path is None:
            return None

        return work_directory_path / 'group_data'

    def _extract_json_labels(self, payload):
        """递归提取 JSON 中的标签，并统计标签对应的形状类型。"""
        labels = set()
        label_stats = {}
        polygon_entries = []
        stack = [payload]
        interesting_keys = {'label', 'labels', 'class', 'class_name', 'category', 'category_name', 'name'}
        shape_type_keys = {'shape_type', 'type', 'geometry', 'geometry_type'}

        def _normalize_text(value):
            if isinstance(value, str):
                text = value.strip()
                if text:
                    return text
            return None

        def _points_count_from_value(value):
            if not isinstance(value, list):
                return 0
            return sum(1 for item in value if isinstance(item, (list, tuple)) and len(item) >= 2)

        def _update_stats(label_text, shape_type_text=None, polygon_points_count=0):
            entry = label_stats.setdefault(label_text, {'types': {}, 'polygon_max_points': 0, 'polygon_point_histogram': {}})
            if shape_type_text:
                entry['types'][shape_type_text] = entry['types'].get(shape_type_text, 0) + 1
                if shape_type_text == 'polygon' and polygon_points_count > entry['polygon_max_points']:
                    entry['polygon_max_points'] = polygon_points_count
                if shape_type_text == 'polygon' and polygon_points_count > 0:
                    histogram = entry.setdefault('polygon_point_histogram', {})
                    histogram[polygon_points_count] = histogram.get(polygon_points_count, 0) + 1

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

                if dict_shape_type == 'polygon':
                    dict_polygon_points_count = _points_count_from_value(current.get('points'))

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
                    if dict_shape_type == 'polygon' and dict_polygon_points_count > 0:
                        polygon_entries.append((label_text, dict_polygon_points_count))

            elif isinstance(current, list):
                stack.extend(current)

        return labels, label_stats, polygon_entries

    def _find_related_image_paths(self, json_file_path):
        related_images = []
        for suffix in IMAGE_SUFFIXES:
            candidate = json_file_path.with_suffix(suffix)
            if candidate.exists() and candidate.is_file():
                related_images.append(candidate)
        return related_images

    def _refresh_preview_label_scan_result(self, labels, scanned_file_count=0, json_file_count=0, error_messages=None, label_stats=None):
        """刷新标签扫描预览区，同时同步右侧下拉框和映射表。"""
        error_messages = error_messages or []
        label_stats = label_stats or {}
        sorted_labels = sorted({label.strip() for label in labels if str(label).strip()})

        self._populate_label_mapping_layout(sorted_labels, label_stats=label_stats)

        if hasattr(self, 'NFPlineEdit') and self.NFPlineEdit is not None:
            self.NFPlineEdit.blockSignals(True)
            self.NFPlineEdit.setText(self._get_dominant_polygon_points(label_stats))
            self.NFPlineEdit.blockSignals(False)

        if hasattr(self, 'comboBoxPreviewZeroClass') and self.comboBoxPreviewZeroClass is not None:
            self.comboBoxPreviewZeroClass.blockSignals(True)
            self.comboBoxPreviewZeroClass.clear()
            self.comboBoxPreviewZeroClass.addItems(sorted_labels)
            self.comboBoxPreviewZeroClass.blockSignals(False)

        if not hasattr(self, 'textBrowserPreviewLabelScanResult') or self.textBrowserPreviewLabelScanResult is None:
            return

        if not sorted_labels:
            message_lines = [
                '未找到可用的 JSON 标签。',
                f'扫描文件数: {scanned_file_count}',
                f'JSON 文件数: {json_file_count}',
            ]
            if error_messages:
                message_lines.append('')
                message_lines.append('解析失败文件:')
                message_lines.extend(error_messages)
            self.textBrowserPreviewLabelScanResult.setPlainText('\n'.join(message_lines))
            return

        message_lines = [
            f'扫描文件数: {scanned_file_count}',
            f'JSON 文件数: {json_file_count}',
            f'标签总数: {len(sorted_labels)}',
            '',
            '标签列表:',
        ]
        message_lines.extend(f'- {label}' for label in sorted_labels)

        if error_messages:
            message_lines.append('')
            message_lines.append('解析失败文件:')
            message_lines.extend(error_messages)

        self.textBrowserPreviewLabelScanResult.setPlainText('\n'.join(message_lines))

    def _get_selected_folder_tree_path(self):
        tree_index = self.folderTreeView.currentIndex()
        if not tree_index.isValid():
            return ""

        tree_model = self.folderTreeView.model()
        if tree_model is None or not hasattr(tree_model, "filePath"):
            return ""

        selected_path = str(Path(tree_model.filePath(tree_index)).expanduser()).strip()
        if not selected_path:
            return ""

        if not Path(selected_path).is_dir():
            return ""

        return selected_path

    def _handle_main_tab_changed(self, index):
        """在切换到标签处理页时，提示用户可以重新扫描 group_data。"""
        tab_widget = getattr(self, 'tabWidgetMain', None)
        if tab_widget is None:
            return

        current_widget = tab_widget.widget(index)
        if current_widget is None:
            return

        tab_name = current_widget.objectName()
        if tab_name == 'tabPreviewLabelMap':
            self._append_log_message('已切换到标签处理页。可以点击“扫描所有文件”刷新 group_data 标签。')
        elif tab_name == 'tabVisualTrain':
            self._refresh_visual_train_image_list(force=self._visual_train_dirty)
            if self._visual_train_current_index < 0 and self._visual_train_image_pairs:
                self._visual_train_current_index = 0
            self._show_visual_train_current_image()
        elif tab_name == 'tabInference':
            # 切换到推理与误差分析页时，扫描 workspace 下 runs/pose 和 runs/HRNet 的结果，填充下拉框
            self._refresh_inference_model_lists()

    def _refresh_inference_model_lists(self):
        """扫描 workspace 下的 runs/pose 和 runs/HRNet 文件夹，按时间倒序填充推理页的下拉框。

        - `comboBox` 用于 Yolo，显示 runs/pose 下的子文件夹，最近的在最前面。
        - `comboBox_2` 用于 HRNet，首项为 'None'（表示不使用 Yolo 结果），随后为 runs/HRNet 下的子文件夹，最近的在最前面。
        每个下拉项会把完整路径保存在 itemData 中，便于后续使用。
        """
        work_dir = self._get_work_directory_path()
        # 先清空，确保 UI 可用
        try:
            if hasattr(self, 'YoloModelCombbox') and self.YoloModelCombbox is not None:
                self.YoloModelCombbox.clear()
            if hasattr(self, 'HRNetModelCombbox') and self.HRNetModelCombbox is not None:
                self.HRNetModelCombbox.clear()
        except Exception:
            pass

        if work_dir is None:
            return

        runs_dir = work_dir / 'runs'

        # YOLO: runs/pose
        try:
            yolo_root = runs_dir / 'pose'
            yolo_items = []
            if yolo_root.exists() and yolo_root.is_dir():
                for child in yolo_root.iterdir():
                    if child.is_dir():
                        stat = child.stat()
                        yolo_items.append((child.name, child, stat.st_mtime))
            # 按修改时间倒序
            yolo_items.sort(key=lambda x: x[2], reverse=True)
            if hasattr(self, 'YoloModelCombbox') and self.YoloModelCombbox is not None:
                for name, path_obj, _ in yolo_items:
                    try:
                        self.YoloModelCombbox.addItem(name, str(path_obj))
                    except Exception:
                        # 兼容不同 Qt 版本的签名
                        self.YoloModelCombbox.addItem(name)
        except Exception:
            pass

        # HRNet: runs/HRNet
        try:
            hr_root = runs_dir / 'HRNet'
            hr_items = []
            if hr_root.exists() and hr_root.is_dir():
                for child in hr_root.iterdir():
                    if child.is_dir():
                        stat = child.stat()
                        hr_items.append((child.name, child, stat.st_mtime))
            hr_items.sort(key=lambda x: x[2], reverse=True)
            if hasattr(self, 'HRNetModelCombbox') and self.HRNetModelCombbox is not None:
                # 首项为 None
                self.HRNetModelCombbox.addItem('None', '')
                for name, path_obj, _ in hr_items:
                    try:
                        self.HRNetModelCombbox.addItem(name, str(path_obj))
                    except Exception:
                        self.HRNetModelCombbox.addItem(name)
        except Exception:
            pass

    def _get_datasets_directory(self):
        work_directory_path = self._get_work_directory_path()
        if work_directory_path is None:
            return None
        return work_directory_path / 'datasets'

    def _refresh_visual_train_image_list(self, force=False):
        """
        只收集 folderTreeView 当前选中目录（如 group_data/xxx）下的图片，
        并在 datasets/images/{train,val,test} 下查找 flatten/stem 同名图片和对应 labels。
        """
        if not force and not self._visual_train_dirty and self._visual_train_image_pairs:
            return self._visual_train_image_pairs

        datasets_dir = self._get_datasets_directory()
        image_pairs = []

        if datasets_dir is not None and datasets_dir.exists():
            images_root = datasets_dir / 'images'
            labels_root = datasets_dir / 'labels'
            split_names = ['train', 'val', 'test']
            selected_path = self._get_selected_folder_tree_path()
            if "datasets" in str(selected_path):
                a = 1
                selected_path = self._get_selected_folder_tree_path()
            if not selected_path:
                group_images = []
            else:
                selected_path = Path(selected_path)
                group_images = [p for p in selected_path.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]


            # 直接复用 s1_dataJson2Train.py 的 flatten 规则，保证和划分时一致
            from s1_dataJson2Train import _flatten_rel_path

            group_data_dir = self._get_group_data_directory()
            for img in sorted(group_images):
                # rel 必须是 group_data 根目录的相对路径，和数据划分时完全一致
                try:
                    rel = img.relative_to(group_data_dir)
                    flat_name = rel.name
                except ValueError:
                    continue
                found = False
                for split in split_names:
                    img_candidate = images_root / split / flat_name
                    label_candidate = labels_root / split / (Path(flat_name).with_suffix('.txt'))
                    if img_candidate.exists():
                        image_pairs.append((img_candidate, label_candidate))
                        found = True
                        break
                # 如果 datasets 里没找到就跳过

        self._visual_train_image_pairs = image_pairs
        self._visual_train_dirty = False
        if self._visual_train_current_index >= len(self._visual_train_image_pairs):
            self._visual_train_current_index = len(self._visual_train_image_pairs) - 1
        return self._visual_train_image_pairs

    def _update_visual_train_progress(self):
        if not hasattr(self, 'progressLabel') or self.progressLabel is None:
            return

        total = len(self._visual_train_image_pairs)
        if total == 0:
            self.progressLabel.setText('0 / 0')
            return

        current = self._visual_train_current_index + 1 if self._visual_train_current_index >= 0 else 0
        self.progressLabel.setText(f'{current} / {total}')

    def _show_visual_train_current_image(self):
        if not self._visual_train_image_pairs:
            if hasattr(self, 'progressLabel') and self.progressLabel is not None:
                self.progressLabel.setText('0 / 0')
            if hasattr(self, '_visual_train_image_view') and self._visual_train_image_view is not None:
                self._visual_train_image_view.clear()
            return

        self._visual_train_current_index = max(0, min(self._visual_train_current_index, len(self._visual_train_image_pairs) - 1))
        image_path, label_path = self._visual_train_image_pairs[self._visual_train_current_index]
        visual_image = visual_Yolo_trainData(str(image_path), str(label_path))

        if hasattr(self, '_visual_train_image_view') and self._visual_train_image_view is not None:
            self._visual_train_image_view.SetImage(visual_image)

        self._update_visual_train_progress()

    def _show_visual_train_image(self, index):
        if not self._refresh_visual_train_image_list():
            self._show_visual_train_current_image()
            return

        if not self._visual_train_image_pairs:
            self._show_visual_train_current_image()
            return

        self._visual_train_current_index = index % len(self._visual_train_image_pairs)
        self._show_visual_train_current_image()

    def _show_error_histogram(self, image_path):
        """在主线程中把误差直方图挂到误差可视化区域。"""
        try:
            if hasattr(self, 'labelErrorVisPlaceholder') and self.labelErrorVisPlaceholder is not None:
                self.labelErrorVisPlaceholder.setVisible(False)
        except Exception:
            pass

        try:
            if hasattr(self, 'errorVisView') and self.errorVisView is not None:
                self.errorVisView.setVisible(False)
        except Exception:
            pass

        if not hasattr(self, '_error_image_view') or self._error_image_view is None:
            parent_widget = self.visualErrGroupBox if hasattr(self, 'visualErrGroupBox') else None
            self._error_image_view = ImageView(parent_widget)
            self._error_image_view.setObjectName('errorImageView')

            if hasattr(self, 'verticalLayout_errorVis') and self.verticalLayout_errorVis is not None:
                self.verticalLayout_errorVis.addWidget(self._error_image_view)
            elif parent_widget is not None:
                layout = parent_widget.layout()
                if layout is not None:
                    layout.addWidget(self._error_image_view)

        self._error_image_view.setVisible(True)
        self._error_image_view.SetImage(str(image_path))

    def visual_train_last_image_slot(self):
        self._refresh_visual_train_image_list()
        if not self._visual_train_image_pairs:
            self._show_visual_train_current_image()
            return

        if self._visual_train_current_index < 0:
            self._visual_train_current_index = 0
        else:
            self._visual_train_current_index = (self._visual_train_current_index - 1) % len(self._visual_train_image_pairs)
        self._show_visual_train_current_image()

    def visual_train_next_image_slot(self):
        self._refresh_visual_train_image_list()
        if not self._visual_train_image_pairs:
            self._show_visual_train_current_image()
            return

        if self._visual_train_current_index < 0:
            self._visual_train_current_index = 0
        else:
            self._visual_train_current_index = (self._visual_train_current_index + 1) % len(self._visual_train_image_pairs)
        self._show_visual_train_current_image()

    def split_TrainData_slot(self):
        """将 group_data 目录里的图片和 JSON 划分为训练数据 """
        group_data_directory = self._get_group_data_directory()
        if group_data_directory is None:
            self._append_log_message('请先选择有效的工作目录。')
            return

        self._append_log_message(f'当前数据划分预览目录: {group_data_directory}')
        if not group_data_directory.exists():
            self._append_log_message('group_data 目录还不存在，请先生成数据。')
            return

        json_file_count = sum(1 for _ in group_data_directory.rglob('*.json'))
        image_file_count = sum(1 for path in group_data_directory.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        self._append_log_message(f'group_data 统计: {image_file_count} 张图片，{json_file_count} 个 JSON 文件。')

        convert_info = self._build_train_split_convert_info(group_data_directory)
        if convert_info is None:
            return

        result_holder = {'error': None, 'stdout': '', 'train_files': None, 'val_files': None, 'test_files': None}

        def _run_split_process():
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                    train_files, val_files, test_files = process_filesYolo(convert_info)
                    process_filesHRNet(convert_info)
                result_holder['train_files'] = train_files
                result_holder['val_files'] = val_files
                result_holder['test_files'] = test_files
            except Exception as exc:
                result_holder['error'] = exc
            finally:
                result_holder['stdout'] = buffer.getvalue()

        self._append_log_message(
            '开始划分数据: '
            f"Append={convert_info.Append}, TrainRatio={convert_info.TrainRatio}, "
            f"ValRatio={convert_info.ValRatio}, TestRatio={convert_info.TestRatio}, "
            f"Seed={convert_info.Seed}, NFP={convert_info.NFP}"
        )

        worker = threading.Thread(target=_run_split_process, daemon=True)
        worker.start()

        while worker.is_alive():
            QApplication.processEvents()
            time.sleep(0.2)

        worker.join()

        if result_holder['stdout']:
            for line in result_holder['stdout'].splitlines():
                if line.strip():
                    self._append_log_message(line)

        if result_holder['error'] is not None:
            self._append_log_message(f'数据划分失败: {result_holder["error"]}')
            return

        train_files = result_holder['train_files'] or []
        val_files = result_holder['val_files'] or []
        test_files = result_holder['test_files'] or []
        self._append_log_message(
            '数据划分完成: '
            f'train={len(train_files)}，val={len(val_files)}，test={len(test_files)}，'
            f'输出目录: {convert_info.DatasetsDir}'
        )
        self._visual_train_dirty = True

    def _build_train_split_convert_info(self, group_data_directory):
        """从界面读取划分参数并构造 s1_dataJson2Train 使用的配置对象。"""
        work_directory_path = self._get_work_directory_path()
        if work_directory_path is None:
            self._append_log_message('请先选择有效的工作目录。')
            return None

        convert_info = ConvertInfo()
        convert_info.JsonPath = str(group_data_directory)
        convert_info.DatasetsDir = str(work_directory_path / 'datasets')

        write_mode_combo = getattr(self, 'comboBoxPreviewWriteMode', None)
        convert_info.Append = bool(write_mode_combo is not None and write_mode_combo.currentIndex() == 1)

        train_ratio_widget = getattr(self, 'doubleSpinBoxPreviewTrainRatio', None)
        val_ratio_widget = getattr(self, 'doubleSpinBoxPreviewValRatio', None)
        test_ratio_widget = getattr(self, 'doubleSpinBoxPreviewTestRatio', None)
        seed_widget = getattr(self, 'spinBoxPreviewSeed', None)
        nfp_widget = getattr(self, 'NFPlineEdit', None)

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
                self._append_log_message('NFP 输入无效，已回退到默认值 4。')

        convert_info.Label2Int = self._collect_label2int_mapping()
        convert_info.OccupiedLabel = self.collect_occupiedLabel()
        if not convert_info.Label2Int:
            self._append_log_message('没有读取到可用的标签映射，请先扫描并设置“训练时ID”。')
            return None

        return convert_info

    def _collect_label2int_mapping(self):
        """把标签映射表和用途选择转换成 s1 所需的 Label2Int。"""
        mapping = {}
        zero_class_label = ''
        zero_class_combo = getattr(self, 'comboBoxPreviewZeroClass', None)
        if zero_class_combo is not None:
            zero_class_label = zero_class_combo.currentText().strip()

        for label_text, line_edit in self._label_id_edits.items():
            if line_edit is None:
                continue

            usage_combo = self._label_usage_combos.get(label_text)
            if usage_combo is not None:
                usage_text = usage_combo.currentText().strip()
                if usage_text and usage_text not in ('请选择用途', '用于训练'):
                    continue

            raw_value = line_edit.text().strip()
            if not raw_value:
                continue

            try:
                mapping[label_text] = int(raw_value)
            except ValueError:
                self._append_log_message(f'标签 {label_text} 的训练时ID无效，已忽略: {raw_value}')

        if zero_class_label:
            mapping[zero_class_label] = 0

        return mapping

    def collect_occupiedLabel(self):
        """收集所有被标记为‘用于遮挡点’的 label 的 rectangle 标注，返回 OccupiedLabel 列表。"""
        occupied_rects = []
        for label_text, usage_combo in self._label_usage_combos.items():
            if usage_combo is None:
                continue
            usage_text = usage_combo.currentText().strip()
            if usage_text != '用于遮挡点':
                continue
            # 假设 self._label_occupied_rects[label_text] 是 rectangle 标注的列表
            occupied_rects.append(label_text)
        return occupied_rects

    def scan_group_data_labels_slot(self):
        """扫描 group_data 下的 JSON 标签，并汇总异常的 polygon 点数。"""
        group_data_directory = self._get_group_data_directory()
        if group_data_directory is None:
            self._append_log_message('请先选择有效的工作目录。')
            self._refresh_preview_label_scan_result([])
            return

        if not group_data_directory.exists():
            self._append_log_message(f'未找到 group_data 目录: {group_data_directory}')
            self._refresh_preview_label_scan_result([], error_messages=[])
            return

        all_files = [path for path in group_data_directory.rglob('*') if path.is_file()]
        json_files = sorted(path for path in all_files if path.suffix.lower() == '.json')
        if not json_files:
            self._append_log_message(f'在 {group_data_directory} 中没有找到 JSON 文件。')
            self._refresh_preview_label_scan_result([], scanned_file_count=len(all_files), json_file_count=0)
            return

        labels = set()
        label_stats = {}
        polygon_point_histogram = {}
        polygon_records = []
        error_messages = []

        for json_file in json_files:
            try:
                with json_file.open('r', encoding='utf-8-sig') as file:
                    payload = json.load(file)
            except (OSError, json.JSONDecodeError) as exc:
                error_messages.append(json_file.name)
                self._append_log_message(f'解析 JSON 失败: {json_file} ({exc})')
                continue

            extracted_labels, extracted_stats, polygon_entries = self._extract_json_labels(payload)
            labels.update(extracted_labels)
            for label_text, stats in extracted_stats.items():
                current_stats = label_stats.setdefault(label_text, {'types': {}, 'polygon_max_points': 0, 'polygon_point_histogram': {}})
                for shape_type, count in (stats.get('types') or {}).items():
                    current_stats['types'][shape_type] = current_stats['types'].get(shape_type, 0) + count
                current_stats['polygon_max_points'] = max(
                    current_stats['polygon_max_points'],
                    int(stats.get('polygon_max_points') or 0),
                )
                for points_count, count in (stats.get('polygon_point_histogram') or {}).items():
                    current_stats['polygon_point_histogram'][points_count] = current_stats['polygon_point_histogram'].get(points_count, 0) + count

            for label_text, points_count in polygon_entries:
                label_hist = polygon_point_histogram.setdefault(label_text, {})
                label_hist[points_count] = label_hist.get(points_count, 0) + 1
                polygon_records.append(
                    {
                        'label': label_text,
                        'points': points_count,
                        'json_file': json_file,
                    }
                )

        # 找到异常polygon点数图片
        anomaly_logs = []
        for label_text, points_hist in polygon_point_histogram.items():
            if len(points_hist) <= 1:
                continue

            majority_points, majority_count = max(points_hist.items(), key=lambda item: (item[1], item[0]))
            if majority_count <= 1:
                continue

            for record in polygon_records:
                if record['label'] != label_text:
                    continue
                if record['points'] == majority_points:
                    continue

                related_images = self._find_related_image_paths(record['json_file'])
                target_files = related_images if related_images else [record['json_file']]
                target_text = ', '.join(str(path) for path in target_files)
                anomaly_logs.append(
                    f'异常图片: label={label_text}, 主流点数={majority_points}, 当前点数={record["points"]}, 文件={target_text}'
                )

        if anomaly_logs:
            self._append_log_message('检测到 polygon 点数异常:')
            for message in anomaly_logs:
                self._append_log_message(message)

        self._refresh_preview_label_scan_result(
            labels,
            scanned_file_count=len(all_files),
            json_file_count=len(json_files),
            error_messages=error_messages,
            label_stats=label_stats,
        )
        self._append_log_message(
            f'扫描完成: {len(all_files)} 个文件，其中 {len(json_files)} 个 JSON 文件，找到 {len(labels)} 个标签。'
        )

    def data_processing_slot(self):
        """在后台执行数据处理，并把进度实时输出到日志区。"""
        source_folder_text = self._get_selected_folder_tree_path()
        work_directory_text = self.workDirectoryLineEdit.currentText().strip()

        if not source_folder_text:
            self._append_log_message('请先在树中选择源文件夹。')
            return

        if not work_directory_text:
            self._append_log_message('请先选择工作目录。')
            return

        source_folder = Path(source_folder_text)
        work_directory_path = Path(work_directory_text)
        output_folder = work_directory_path / 'group_data'
        group_size = self.spinBoxGroupSize.value()
        total_file_count = sum(1 for _ in find_images(source_folder))

        result_holder = {'result': None, 'error': None}

        def _run_process_data():
            try:
                result_holder['result'] = process_data(source_folder, output_folder, group_size)
            except Exception as exc:
                result_holder['error'] = exc

        # 记录已有的数据数量
        N = sum(1 for path in output_folder.rglob('*') if path.is_file())
        worker = threading.Thread(target=_run_process_data, daemon=True)
        worker.start()

        last_file_count = -1
        while worker.is_alive():
            current_file_count = 0
            if output_folder.exists():
                current_file_count = sum(1 for path in output_folder.rglob('*') if path.is_file())

            if current_file_count != last_file_count:
                self._append_log_message(
                    f'process_data 进度: {current_file_count-N} / 总文件数 {total_file_count}'
                )
                last_file_count = current_file_count

            QApplication.processEvents()
            time.sleep(0.2)

        worker.join()

        if result_holder['error'] is not None:
            self._append_log_message(f'数据处理失败: {result_holder["error"]}')
            return

        result = result_holder['result']

        self._append_log_message(
            '数据处理完成: '
            f"共找到 {result['image_count']} 张图片，生成 {result['group_count']} 个分组，"
            f"输出目录: {result['output_folder']}"
        )


    def choose_working_directory_slot(self, target_line_edit=None):
        """弹出文件夹选择框，让用户切换当前工作目录。"""
        start_directory = ""
        if self.workDirectoryLineEdit is not None:
            start_directory = self.workDirectoryLineEdit.currentText().strip()

        if not start_directory:
            start_directory = str(Path.home())

        selected_directory = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            start_directory,
        )

        if not selected_directory:
            return

        self._set_work_directory(selected_directory)

    def batch_infer_slot(self):
        """批量推理槽：在后台加载模型并对指定文件夹下的所有图片进行推理，保存结果并输出到日志。"""
        
        label_map = self.get_label_id_mapping()
        if not label_map:
            self._append_log_message("请先扫描并填写标签ID映射！")
            return
        
        class_names = {v: k for k, v in label_map.items()}
       
        import traceback
        try:
            from s4_inference import InferenceModel, draw_results, save_result, statistics_result
        except Exception as e:
            self._append_log_message(f'无法导入 s4_inference: {e}')
            return

        work_dir = self._get_work_directory_path()
        if work_dir is None:
            self._append_log_message('请先选择有效的工作目录。')
            return

        # 输入图片文件夹：优先使用推理页的输入框，否则使用 work_dir/datasets/images/test
        infer_folder_text = ''
        if hasattr(self, 'inferImgForlderLineEdit') and self.inferImgForlderLineEdit is not None:
            infer_folder_text = self.inferImgForlderLineEdit.text().strip()
        
        if infer_folder_text == '':
            self._append_log_message('请先选择批处理图片目录。')
            return

        if not infer_folder_text:
            infer_folder = work_dir / 'datasets' / 'images' / 'test'
        else:
            infer_folder = Path(infer_folder_text).expanduser()

        if not infer_folder.exists() or not infer_folder.is_dir():
            self._append_log_message(f'未找到推理输入目录: {infer_folder}')
            return

        # 选择 YOLO 和 HRNet 模型路径（下拉项的 itemData 存储路径）
        yolo_model_path = None
        if hasattr(self, 'YoloModelCombbox') and self.YoloModelCombbox is not None:
            try:
                yolo_model_path = self.YoloModelCombbox.currentData()
            except Exception:
                yolo_model_path = None

        hrnet_model_path = None
        hrnet_model_path = self._get_selected_model_path_from_combo('HRNetModelCombbox')

        if not yolo_model_path:
            self._append_log_message('请在推理页选择一个 YOLO 模型（runs/pose 下的子目录）。')
            return

        # 输出目录可以通过 UI 指定（batchInferRetFolderLineEdit），否则回退到 work_dir/inference
        ret_folder_text = ''
        if hasattr(self, 'batchInferRetFolderLineEdit') and self.batchInferRetFolderLineEdit is not None:
            try:
                ret_folder_text = self.batchInferRetFolderLineEdit.text().strip()
            except Exception:
                ret_folder_text = ''

        base_ret_dir = work_dir / ret_folder_text
        shutil.rmtree(base_ret_dir, ignore_errors=True)  # 清空旧结果
        out_json_dir = base_ret_dir / 'json'
        out_vis_dir = base_ret_dir / 'vis'
        out_json_dir.mkdir(parents=True, exist_ok=True)
        out_vis_dir.mkdir(parents=True, exist_ok=True)

        model = InferenceModel()
        result_holder = {'combined_png': None}

        def _run_batch():
            try:
                self._append_log_message(f'加载 YOLO 模型: {yolo_model_path}')
                model.load_yolo_model(str(yolo_model_path+"/weights/best.pt"))
                if hrnet_model_path:
                    try:
                        self._append_log_message(f'加载 HRNet 模型: {hrnet_model_path}')
                        model.load_hrnet_model(str(hrnet_model_path))
                    except Exception as exc:
                        self._append_log_message(f'加载 HRNet 失败，继续仅使用 YOLO: {exc}')

                images = [p for p in sorted(infer_folder.rglob('*')) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
                total = len(images)
                if total == 0:
                    self._append_log_message(f'未在 {infer_folder} 找到图片。')
                    return

                self._append_log_message(f'开始对 {total} 张图片进行推理...')

                # 根据 UI 复选框决定是否保存 JSON、可视化图片或统计误差
                save_json_enabled = False
                save_img_enabled = False
                save_err_enabled = False
                try:
                    if hasattr(self, 'saveJsonCB') and self.saveJsonCB is not None:
                        save_json_enabled = bool(self.saveJsonCB.isChecked())
                except Exception:
                    save_json_enabled = False

                try:
                    if hasattr(self, 'saveImgCB') and self.saveImgCB is not None:
                        save_img_enabled = bool(self.saveImgCB.isChecked())
                except Exception:
                    save_img_enabled = False

                try:
                    if hasattr(self, 'saveErrCB') and self.saveErrCB is not None:
                        save_err_enabled = bool(self.saveErrCB.isChecked())
                except Exception:
                    save_err_enabled = False

                # collect ground-truth items and prediction results for later error statistics
                gt_files = []
                pred_ret = []

                for idx, img_path in enumerate(images, start=1):
                    try:
                        ret = model.predict(str(img_path))
                        # collect prediction
                        pred_ret.append(ret)
                        # try to locate ground-truth file for this image
                        gt_item = None
                        try:
                            # find corresponding gt .txt under datasets by image stem
                            dataset_dir = Path(work_dir) / 'datasets'
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

                            gt_item = txt_path
                        except Exception:
                            gt_item = None
                        gt_files.append(gt_item)
                        json_path = out_json_dir / (img_path.stem + '.json')

                        if save_json_enabled:
                            try:
                                save_result(str(img_path), img_path.name, ret, str(json_path), class_names=class_names)
                            except Exception:
                                pass

                        if save_img_enabled:
                            try:
                                vis = draw_results(str(img_path), ret, class_names=class_names)
                                vis_path = out_vis_dir / (img_path.stem + '.png')
                                try:
                                    import cv2
                                    cv2.imwrite(str(vis_path), vis)
                                except Exception:
                                    pass
                            except Exception:
                                pass

                        if idx % 10 == 0 or idx == total:
                            self._append_log_message(f'推理进度: {idx} / {total}')
                    except Exception as exc:
                        self._append_log_message(f'处理文件 {img_path} 失败: {exc}')

                # 如果用户勾选了统计误差，调用 s4_inference.statistics_result
                if save_err_enabled:
                    try:
                        self._append_log_message('开始统计误差...')
                        # stats runs may be expensive; run synchronously here but catch errors
                        # use the previously constructed base_ret_dir for result paths
                        statistics_result(gt_files, pred_ret, class_names, str(base_ret_dir))
                        self._append_log_message('误差统计完成。')
                        hist_dir = Path(base_ret_dir) / 'error_hist'
                        combined_png = hist_dir / 'combined.png'
                        if not combined_png.exists() and hist_dir.exists():
                            pngs = sorted(hist_dir.glob('*.png'))
                            combined_png = pngs[0] if pngs else combined_png

                        if combined_png.exists():
                            result_holder['combined_png'] = combined_png
                    except Exception as exc:
                        self._append_log_message(f'误差统计失败: {exc}')

                self._append_log_message(f'批量推理完成，结果保存在: {out_json_dir} 与 {out_vis_dir}')
            except Exception as exc:
                self._append_log_message(f'批量推理失败: {exc}\n{traceback.format_exc()}')

        worker = threading.Thread(target=_run_batch, daemon=True)
        worker.start()

        while worker.is_alive():
            QApplication.processEvents()
            time.sleep(0.2)

        worker.join()

        combined_png = result_holder.get('combined_png')
        if combined_png is not None:
            try:
                self._show_error_histogram(combined_png)
            except Exception as exc:
                self._append_log_message(f'显示直方图失败: {exc}')


def main():
    """程序入口：创建应用并显示主窗口。"""
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = VisualTrainFactoryWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
