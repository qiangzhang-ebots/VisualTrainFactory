from __future__ import annotations

import shutil
import traceback
from pathlib import Path

from ImageView import ImageView
from app.constants import IMAGE_SUFFIXES, LabelUsage, ModelKind, Split, current_model_task
from app.controllers.base import TabController
from app.qt_imports import QAbstractItemView, QCheckBox, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget
from app.task_runner import run_thread_with_process_events
from app.widget_io import read_bool, read_combo_data, read_text

# 利用训练好的模型，进行推理，并且可视化结果
class InferenceController(TabController):
    def __init__(self, window):
        super().__init__(window)
        self.save_part_label_panel = None
        self.save_part_label_list = None

    def configure_ui(self):
        group = getattr(self.window, "batchInferSettingGroupBox", None)
        if group is None:
            return
        layout = group.layout()
        if layout is None:
            return

        panel = QWidget(group)
        panel.setObjectName("savePartLabelPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 4, 8, 4)
        panel_layout.setSpacing(4)

        label = QLabel("选择要保存的标签（仅在保存部分标签时生效）", panel)
        label.setStyleSheet("color: #cbd5e1;")
        panel_layout.addWidget(label)

        list_widget = QListWidget(panel)
        list_widget.setSelectionMode(QAbstractItemView.NoSelection)
        try:
            list_widget.setSpacing(1)
        except Exception:
            pass
        panel_layout.addWidget(list_widget)
        panel.setVisible(False)

        idx = layout.indexOf(self.window.btnBatchInfer)
        if idx >= 0:
            layout.insertWidget(idx, panel)
        else:
            layout.addWidget(panel)

        self.save_part_label_panel = panel
        self.save_part_label_list = list_widget

    def on_enter(self):
        self.sync_task_ui()

    def sync_task_ui(self):
        task = current_model_task(self.window)
        is_pose = task == "pose"
        for widget_name in ("labelHrnetModel", "comboBoxHrnetModel", "btnExportHrnetOnnx"):
            widget = getattr(self.window, widget_name, None)
            if widget is not None:
                widget.setEnabled(is_pose)
        self.refresh_model_lists()

    def on_folder_tree_changed(self, selected, deselected):
        try:
            selected_path = self.window.get_selected_folder_tree_path()
            if selected_path and hasattr(self.window, "lineEditInferImageFolder"):
                self.window.lineEditInferImageFolder.blockSignals(True)
                self.window.lineEditInferImageFolder.setText(selected_path)
                self.window.lineEditInferImageFolder.blockSignals(False)
        except Exception:
            pass

    def export_yolo_onnx(self):
        try:
            from s5_exportOnnx import exportYoloOnnx

            if self.get_workspace() is None:
                self.append_log("请先选择有效的工作目录。")
                return

            yolo_model_path = read_combo_data(self.window, "comboBoxYoloModel")
            if not yolo_model_path:
                self.append_log("请先在推理页选择一个 YOLO 模型。")
                return

            self.append_log(f"开始导出 YOLO ONNX: {yolo_model_path}")
            exportYoloOnnx(yolo_model_path)
            self.append_log("YOLO ONNX 导出完成")
        except Exception as exc:
            self.append_log(f"YOLO ONNX 导出失败: {exc}\n{traceback.format_exc()}")

    def export_hrnet_onnx(self):
        try:
            if current_model_task(self.window) != "pose":
                self.append_log("当前任务不加载 HRNet（仅关键点任务可用）。")
                return

            from s5_exportOnnx import exportHRNetOnnx

            if self.get_workspace() is None:
                self.append_log("请先选择有效的工作目录。")
                return

            hrnet_model_path = read_combo_data(self.window, "comboBoxHrnetModel")
            if not hrnet_model_path:
                self.append_log("请先在推理页选择一个 HRNet 模型。")
                return

            self.append_log(f"开始导出 HRNet ONNX: {hrnet_model_path}")
            onnx_path = exportHRNetOnnx(hrnet_model_path)
            self.append_log(f"HRNet ONNX 导出完成: {onnx_path}")
        except Exception as exc:
            self.append_log(f"HRNet ONNX 导出失败: {exc}\n{traceback.format_exc()}")

    def export_engine(self, model_type="yolo"):
        try:
            import subprocess

            if self.get_workspace() is None:
                self.append_log("请先选择有效的工作目录。")
                return

            if model_type == "hrnet":
                if current_model_task(self.window) != "pose":
                    self.append_log("当前任务不加载 HRNet（仅关键点任务可用）。")
                    return

                model_path = read_combo_data(self.window, "comboBoxHrnetModel")
                if not model_path:
                    self.append_log("请先在推理页选择一个 HRNet 模型。")
                    return

                model_dir = Path(model_path)
                # HRNet 导出的 onnx 文件名不固定（如 epoch_200.onnx），且位于模型目录直接下级。
                # 这里直接扫描目录下的 *.onnx，选取最新生成的一个。
                onnx_files = sorted(
                    model_dir.glob("*.onnx"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if not onnx_files:
                    self.append_log(
                        f"未在模型目录找到 onnx 文件: {model_dir}，请先执行“导出onnx模型”。"
                    )
                    return
                onnx_path = onnx_files[0]
                self.append_log(f"使用 HRNet onnx: {onnx_path}")
            else:
                model_path = read_combo_data(self.window, "comboBoxYoloModel")
                if not model_path:
                    self.append_log("请先在推理页选择一个 YOLO 模型。")
                    return

                onnx_path = Path(model_path) / "weights" / "best.onnx"

            if not onnx_path.exists():
                self.append_log(
                    f"未找到 onnx 文件: {onnx_path}，请先执行对应的“导出onnx模型”。"
                )
                return

            tensorrt_path = read_text(self.window, "lineEditTensorrtPath").strip()
            if not tensorrt_path:
                self.append_log("请先配置 TensorRT 路径。")
                return

            trtexec_path = Path(tensorrt_path) / "trtexec"
            if not trtexec_path.is_file():
                self.append_log(f"未找到 trtexec 可执行文件: {trtexec_path}")
                return

            engine_path = onnx_path.with_suffix(".engine")
            cmd = [
                str(trtexec_path),
                f"--onnx={onnx_path}",
                f"--saveEngine={engine_path}",
            ]
            self.append_log(f"开始导出 Engine 模型，执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=str(Path(tensorrt_path)),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                self.append_log(
                    f"trtexec 执行失败 (返回码 {result.returncode}):\n{result.stdout}\n{result.stderr}"
                )
                return

            self.append_log(f"Engine 模型导出完成: {engine_path}")
        except Exception as exc:
            self.append_log(f"Engine 模型导出失败: {exc}\n{traceback.format_exc()}")

    def select_tensorrt_path(self):
        from PyQt5.QtWidgets import QFileDialog

        start_directory = read_text(self.window, "lineEditTensorrtPath").strip()
        if not start_directory:
            start_directory = str(Path.home())

        selected_directory = QFileDialog.getExistingDirectory(
            self.window, "选择 TensorRT bin 路径", start_directory
        )
        if not selected_directory:
            return

        self.window.lineEditTensorrtPath.blockSignals(True)
        self.window.lineEditTensorrtPath.setText(selected_directory)
        self.window.lineEditTensorrtPath.blockSignals(False)

    def refresh_model_lists(self):
        workspace = self.get_workspace()
        yolo_combo = getattr(self.window, "comboBoxYoloModel", None)
        hrnet_combo = getattr(self.window, "comboBoxHrnetModel", None)

        try:
            if yolo_combo is not None:
                yolo_combo.clear()
            if hrnet_combo is not None:
                hrnet_combo.clear()
        except Exception:
            pass

        if workspace is None:
            return

        task = current_model_task(self.window)
        is_pose = task == "pose"
        if task == "seg":
            yolo_kind = ModelKind.SEG
        elif task == "obb":
            yolo_kind = ModelKind.OBB
        else:
            yolo_kind = ModelKind.YOLO
        self._fill_model_combo(yolo_combo, workspace.model_runs(yolo_kind), include_none=False)
        if is_pose:
            self._fill_model_combo(hrnet_combo, workspace.model_runs(ModelKind.HRNET), include_none=True)
        elif hrnet_combo is not None:
            hrnet_combo.addItem("None", "")

    @staticmethod
    def _fill_model_combo(combo_box, root_path: Path, include_none: bool):
        if combo_box is None:
            return

        items = []
        if root_path.exists() and root_path.is_dir():
            for child in root_path.iterdir():
                if child.is_dir():
                    stat = child.stat()
                    items.append((child.name, child, stat.st_mtime))
        items.sort(key=lambda item: item[2], reverse=True)

        if include_none:
            combo_box.addItem("None", "")
        for name, path_obj, _ in items:
            try:
                combo_box.addItem(name, str(path_obj))
            except Exception:
                combo_box.addItem(name)

    def populate_save_part_label_widget(self):
        if self.save_part_label_list is None:
            return
        self.save_part_label_list.clear()
        rows = self.window.label_data_preview_controller.collect_label_mapping_rows()
        for row in rows:
            label_text = str(row.get("label", "")).strip()
            if not label_text:
                continue
            list_item = QListWidgetItem()
            checkbox = QCheckBox(label_text)
            checkbox.setStyleSheet("color: #f8fafc;")
            checked = str(row.get("usage", "")).strip() == LabelUsage.TRAIN.value
            checkbox.setChecked(bool(checked))
            self.save_part_label_list.addItem(list_item)
            self.save_part_label_list.setItemWidget(list_item, checkbox)

    def get_save_part_label_ids(self):
        ids = set()
        if self.save_part_label_list is None:
            return ids
        label_map = self.window.label_data_preview_controller.get_label_id_mapping() or {}
        for idx in range(self.save_part_label_list.count()):
            item = self.save_part_label_list.item(idx)
            if item is None:
                continue
            checkbox = self.save_part_label_list.itemWidget(item)
            if checkbox is None or not checkbox.isChecked():
                continue
            label_text = str(checkbox.text()).strip()
            if label_text in label_map:
                try:
                    ids.add(int(label_map[label_text]))
                except (TypeError, ValueError):
                    continue
        return ids

    def collect_save_part_label_checked(self):
        checked_labels = []
        if self.save_part_label_list is None:
            return checked_labels
        for idx in range(self.save_part_label_list.count()):
            item = self.save_part_label_list.item(idx)
            if item is None:
                continue
            checkbox = self.save_part_label_list.itemWidget(item)
            if checkbox is None:
                continue
            try:
                if bool(checkbox.isChecked()):
                    label_text = str(checkbox.text()).strip()
                    if label_text:
                        checked_labels.append(label_text)
            except Exception:
                continue
        return checked_labels

    def restore_save_part_label_panel(self, saved_widget_state=None):
        panel = self.save_part_label_panel
        checkbox = getattr(self.window, "checkBoxSavePartJson", None)
        if checkbox is None or panel is None:
            if panel is not None:
                panel.setVisible(False)
            return

        panel.setVisible(bool(checkbox.isChecked()))
        if checkbox.isChecked():
            self.populate_save_part_label_widget()
            saved_checks = []
            if isinstance(saved_widget_state, dict):
                saved_checks = saved_widget_state.get("__save_part_label_checked__", [])
            if saved_checks and self.save_part_label_list is not None:
                for idx in range(self.save_part_label_list.count()):
                    item = self.save_part_label_list.item(idx)
                    if item is None:
                        continue
                    widget = self.save_part_label_list.itemWidget(item)
                    if widget is None:
                        continue
                    label_text = str(widget.text()).strip()
                    widget.setChecked(bool(label_text in saved_checks))

    def run_batch_inference(self):
        task = current_model_task(self.window)
        use_seg = task == "seg"
        use_obb = task == "obb"
        use_pose = task == "pose"

        label_map = self.window.label_data_preview_controller.get_label_id_mapping()
        if not label_map:
            self.append_log("请先扫描并填写标签ID映射！")
            return

        class_names = {value: key for key, value in label_map.items()}

        try:
            from s4_inference import (
                InferenceModel,
                draw_results,
                save_result,
                statistics_result,
                statistics_result_seg,
            )
        except Exception as exc:
            self.append_log(f"无法导入 s4_inference: {exc}")
            return

        workspace = self.get_workspace()
        if workspace is None:
            self.append_log("请先选择有效的工作目录。")
            return

        infer_folder_text = read_text(self.window, "lineEditInferImageFolder")
        if not infer_folder_text:
            self.append_log("请先选择批处理图片目录。")
            return

        infer_folder = Path(infer_folder_text).expanduser()
        if not infer_folder.exists() or not infer_folder.is_dir():
            self.append_log(f"未找到推理输入目录: {infer_folder}")
            return

        yolo_model_path = read_combo_data(self.window, "comboBoxYoloModel")
        hrnet_model_path = read_combo_data(self.window, "comboBoxHrnetModel")

        if not yolo_model_path:
            runs_dir = "runs/seg" if use_seg else ("runs/obb" if use_obb else "runs/pose")
            self.append_log(f"请在推理页选择一个 YOLO 模型（{runs_dir} 下的子目录）。")
            return

        ret_folder_text = read_text(self.window, "batchInferRetFolderLineEdit")
        base_ret_dir = workspace.root / ret_folder_text
        shutil.rmtree(base_ret_dir, ignore_errors=True)
        out_json_dir = base_ret_dir / "json"
        out_vis_dir = base_ret_dir / "vis"
        out_json_dir.mkdir(parents=True, exist_ok=True)
        out_vis_dir.mkdir(parents=True, exist_ok=True)

        save_json_enabled = read_bool(self.window, "checkBoxSaveJson")
        save_part_json_enabled = read_bool(self.window, "checkBoxSavePartJson")
        save_img_enabled = read_bool(self.window, "checkBoxSaveImage")
        save_err_enabled = read_bool(self.window, "checkBoxSaveError")
        normalize_enabled = read_bool(self.window, "checkBoxNormalizeImage")
        part_ids = self.get_save_part_label_ids() if save_part_json_enabled else set()

        pose_format = None
        keypoint_names = None
        if use_pose:
            from s4_inference import _resolve_pose_format

            pose_format, keypoint_names = _resolve_pose_format(workspace.root)

        model = InferenceModel()
        result_holder = {
            "combined_png": None,
            "gt_files": None,
            "pred_ret": None,
            "error": None,
        }

        def _run_batch():
            try:
                weights_path = str(Path(yolo_model_path) / "weights" / "best.pt")
                task_name = "YOLO 分割" if use_seg else ("YOLO OBB" if use_obb else "YOLO")
                self.append_log(f"加载 {task_name} 模型: {weights_path}")
                model.load_yolo_model(weights_path)
                if use_pose and hrnet_model_path:
                    try:
                        self.append_log(f"加载 HRNet 模型: {hrnet_model_path}")
                        model.load_hrnet_model(str(hrnet_model_path))
                    except Exception as exc:
                        self.append_log(f"加载 HRNet 失败，继续仅使用 YOLO: {exc}")
                else:
                    self.append_log("当前任务不加载 HRNet。")

                images = [
                    path
                    for path in sorted(infer_folder.rglob("*"))
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                ]
                total = len(images)
                if total == 0:
                    self.append_log(f"未在 {infer_folder} 找到图片。")
                    return

                self.append_log(f"开始对 {total} 张图片进行{task_name}推理...")
                gt_files = []
                pred_ret = []

                for idx, img_path in enumerate(images, start=1):
                    try:
                        ret = model.predict(str(img_path), normalize=normalize_enabled)
                        pred_ret.append(ret)
                        gt_item = workspace.find_label_file(img_path.stem)
                        gt_files.append(gt_item)
                        json_path = out_json_dir / (img_path.stem + ".json")

                        if save_json_enabled:
                            try:
                                save_result(
                                    str(img_path), img_path.name, ret, str(json_path), class_names=class_names,
                                    pose_format=pose_format, keypoint_names=keypoint_names,
                                )
                            except Exception:
                                pass

                        if save_part_json_enabled and part_ids:
                            try:
                                save_result(
                                    str(img_path),
                                    img_path.name,
                                    ret,
                                    str(json_path),
                                    class_names=class_names,
                                    part_labels=part_ids,
                                    pose_format=pose_format,
                                    keypoint_names=keypoint_names,
                                )
                            except Exception:
                                pass

                        if save_img_enabled:
                            try:
                                vis = draw_results(str(img_path), ret, class_names=class_names)
                                vis_path = out_vis_dir / (img_path.stem + ".png")
                                try:
                                    import cv2

                                    cv2.imwrite(str(vis_path), vis)
                                except Exception:
                                    pass
                            except Exception:
                                pass

                        if idx % 10 == 0 or idx == total:
                            self.append_log(f"推理进度: {idx} / {total}")
                    except Exception as exc:
                        self.append_log(f"处理文件 {img_path} 失败: {exc}")

                result_holder["gt_files"] = gt_files
                result_holder["pred_ret"] = pred_ret
                self.append_log(f"批量推理完成，结果保存在: {out_json_dir} 与 {out_vis_dir}")
            except Exception as exc:
                result_holder["error"] = f"{exc}\n{traceback.format_exc()}"
                self.append_log(f"批量推理失败: {exc}\n{traceback.format_exc()}")

        run_thread_with_process_events(_run_batch)

        if result_holder.get("error"):
            return

        if (
            save_err_enabled
            and result_holder.get("gt_files") is not None
            and result_holder.get("pred_ret") is not None
        ):
            try:
                self.append_log("开始统计误差...")
                if task == "seg":
                    statistics_result_seg(
                        result_holder["gt_files"],
                        result_holder["pred_ret"],
                        class_names,
                        str(base_ret_dir),
                    )
                else:
                    statistics_result(
                        result_holder["gt_files"],
                        result_holder["pred_ret"],
                        class_names,
                        str(base_ret_dir),
                    )
                self.append_log("误差统计完成。")
                hist_dir = Path(base_ret_dir) / "error_hist"
                combined_png = hist_dir / "combined.png"
                if not combined_png.exists() and hist_dir.exists():
                    pngs = sorted(hist_dir.glob("*.png"))
                    combined_png = pngs[0] if pngs else combined_png

                if combined_png.exists():
                    result_holder["combined_png"] = combined_png
            except Exception as exc:
                self.append_log(f"误差统计失败: {exc}")

        combined_png = result_holder.get("combined_png")
        if combined_png is not None:
            try:
                self.show_error_histogram(combined_png)
            except Exception as exc:
                self.append_log(f"显示直方图失败: {exc}")

    def show_error_histogram(self, image_path):
        if not hasattr(self.window, "error_image_view") or self.window.error_image_view is None:
            parent_widget = self.window.visualErrGroupBox if hasattr(self.window, "visualErrGroupBox") else None
            self.window.error_image_view = ImageView(parent_widget)
            self.window.error_image_view.setObjectName("errorImageView")

            if hasattr(self.window, "verticalLayout_errorVis") and self.window.verticalLayout_errorVis is not None:
                self.window.verticalLayout_errorVis.addWidget(self.window.error_image_view)
            elif parent_widget is not None:
                layout = parent_widget.layout()
                if layout is not None:
                    layout.addWidget(self.window.error_image_view)

        self.window.error_image_view.setVisible(True)
        self.window.error_image_view.SetImage(str(image_path))

    def on_save_json_toggled(self, checked):
        other = getattr(self.window, "checkBoxSavePartJson", None)
        if other is None:
            return
        if checked:
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
        if self.save_part_label_panel is not None and checked:
            self.save_part_label_panel.setVisible(False)

    def on_save_part_json_toggled(self, checked):
        other = getattr(self.window, "checkBoxSaveJson", None)
        if other is None:
            return
        if checked:
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
        if self.save_part_label_panel is not None:
            self.save_part_label_panel.setVisible(bool(checked))
