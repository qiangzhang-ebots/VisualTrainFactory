from __future__ import annotations

from pathlib import Path

from ImageView import ImageView
from app.constants import IMAGE_SUFFIXES, Split
from app.controllers.base import TabController
from s2_visualTrainData import visual_Yolo_trainData

# 数据已经是yolo,hrnet格式，可视化训练集和验证集
class TrainDataVisualizationController(TabController):
    def configure_view(self):
        if not hasattr(self.window, "layoutVisualTrainCanvas"):
            self.window.image_view = None
            return

        self.window.image_view = ImageView(
            self.window.tabVisualTrain if hasattr(self.window, "tabVisualTrain") else self.window
        )
        self.window.image_view.setObjectName("visualTrainImageView")
        self.window.image_view.setMinimumHeight(420)
        self.window.image_view.setStyleSheet("background-color: #000000; border: 1px solid #334155;")
        self.window.layoutVisualTrainCanvas.addWidget(self.window.image_view, 1)

        self.refresh_image_list(force=True)
        self.show_image(0)

    def on_enter(self):
        self.refresh_image_list(force=self.window.is_image_list_dirty)
        if self.window.current_index < 0 and self.window.image_pairs:
            self.window.current_index = 0
        self.show_current_image()

    def refresh_image_list(self, force=False):
        if not force and not self.window.is_image_list_dirty and self.window.image_pairs:
            return self.window.image_pairs

        workspace = self.get_workspace()
        image_pairs = []

        if workspace is not None and workspace.datasets.exists():
            images_root = workspace.datasets / "images"
            labels_root = workspace.datasets / "labels"
            selected_path_text = self.window.get_selected_folder_tree_path()
            if not selected_path_text:
                group_images = []
            else:
                selected_path = Path(selected_path_text)
                group_images = [
                    path
                    for path in selected_path.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                ]

            group_data_dir = workspace.group_data
            for image_path in sorted(group_images):
                try:
                    image_path.relative_to(group_data_dir)
                    flat_name = image_path.name
                except ValueError:
                    continue
                for split in Split:
                    img_candidate = images_root / split.value / flat_name
                    label_candidate = labels_root / split.value / Path(flat_name).with_suffix(".txt")
                    if img_candidate.exists():
                        image_pairs.append((img_candidate, label_candidate))
                        break

        self.window.image_pairs = image_pairs
        self.window.is_image_list_dirty = False
        if self.window.current_index >= len(self.window.image_pairs):
            self.window.current_index = len(self.window.image_pairs) - 1
        return self.window.image_pairs

    def update_progress(self):
        if not hasattr(self.window, "progressLabel") or self.window.progressLabel is None:
            return

        total = len(self.window.image_pairs)
        if total == 0:
            self.window.progressLabel.setText("0 / 0")
            return

        current = self.window.current_index + 1 if self.window.current_index >= 0 else 0
        filename = ""
        if 0 <= self.window.current_index < len(self.window.image_pairs):
            filename = Path(self.window.image_pairs[self.window.current_index][0]).name

        if filename:
            self.window.progressLabel.setText(f"{current} / {total}  {filename}")
        else:
            self.window.progressLabel.setText(f"{current} / {total}")

    def show_current_image(self):
        if not self.window.image_pairs:
            if hasattr(self.window, "progressLabel") and self.window.progressLabel is not None:
                self.window.progressLabel.setText("0 / 0")
            if self.window.image_view is not None:
                self.window.image_view.clear()
            return

        self.window.current_index = max(
            0, min(self.window.current_index, len(self.window.image_pairs) - 1)
        )
        image_path, label_path = self.window.image_pairs[self.window.current_index]
        label_data_preview_controller = getattr(self.window, "label_data_preview_controller", None)
        label_mapping_rows = (
            label_data_preview_controller.saved_label_mapping_rows
            if label_data_preview_controller is not None
            else None
        )
        visual_image = visual_Yolo_trainData(
            str(image_path),
            str(label_path),
            label_mapping_rows=label_mapping_rows,
        )

        if self.window.image_view is not None:
            self.window.image_view.SetImage(visual_image)

        self.update_progress()

    def show_image(self, index):
        if not self.refresh_image_list():
            self.show_current_image()
            return

        if not self.window.image_pairs:
            self.show_current_image()
            return

        self.window.current_index = index % len(self.window.image_pairs)
        self.show_current_image()

    def show_previous_image(self):
        self.refresh_image_list()
        if not self.window.image_pairs:
            self.show_current_image()
            return

        if self.window.current_index < 0:
            self.window.current_index = 0
        else:
            self.window.current_index = (self.window.current_index - 1) % len(self.window.image_pairs)
        self.show_current_image()

    def show_next_image(self):
        self.refresh_image_list()
        if not self.window.image_pairs:
            self.show_current_image()
            return

        if self.window.current_index < 0:
            self.window.current_index = 0
        else:
            self.window.current_index = (self.window.current_index + 1) % len(self.window.image_pairs)
        self.show_current_image()

    def on_folder_tree_changed(self, selected, deselected):
        self.window.mark_image_list_dirty()
        self.window.current_index = -1
        self.refresh_image_list(force=True)
        self.show_current_image()

    def handle_key_press(self, key, modifiers=None, is_auto_repeat=False):
        tab_widget = getattr(self.window, "tabWidgetMain", None)
        if tab_widget is None or not hasattr(self.window, "tabVisualTrain"):
            return False

        current_widget = tab_widget.currentWidget()
        if current_widget is not self.window.tabVisualTrain:
            return False

        from app.qt_imports import Qt

        if modifiers not in (None, Qt.NoModifier):
            return False
        if is_auto_repeat:
            return False

        if key == Qt.Key_A:
            self.show_previous_image()
            return True
        if key == Qt.Key_D:
            self.show_next_image()
            return True
        return False
