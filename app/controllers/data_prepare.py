from __future__ import annotations

from pathlib import Path

from app.controllers.base import TabController
from app.task_runner import run_thread_with_process_events
from app.widget_io import read_bool
from s0_dataprocessing import collect_images, process_data

# 数据准备页，原始数据，划分group，开始标注
class DataPrepareController(TabController):
    def run_data_processing(self):
        source_folder_text = self.window.get_selected_folder_tree_path()
        work_dir_text = self.window.comboBoxWorkDirectory.currentText().strip()

        if not source_folder_text:
            self.append_log("请先在树中选择源文件夹。")
            return

        if not work_dir_text:
            self.append_log("请先选择工作目录。")
            return

        radio_system = getattr(self.window, "radioSystem", None)
        radio_ui = getattr(self.window, "radioUi", None)
        if radio_system is not None and radio_system.isChecked():
            source_mode = "system"
        elif radio_ui is not None and radio_ui.isChecked():
            source_mode = "ui"
        else:
            self.append_log("请选择数据源类型：system 或 ui。")
            return

        cam_checkbox_map = {
            "denali0_cam0": getattr(self.window, "checkDenali0Cam0", None),
            "denali0_cam1": getattr(self.window, "checkDenali0Cam1", None),
            "denali1_cam0": getattr(self.window, "checkDenali1Cam0", None),
            "denali1_cam1": getattr(self.window, "checkDenali1Cam1", None),
        }
        selected_cams = {
            key for key, checkbox in cam_checkbox_map.items()
            if checkbox is not None and checkbox.isChecked()
        }
        if not selected_cams:
            self.append_log("请至少勾选一个相机（denali*_cam*）。")
            return

        source_folder = Path(source_folder_text)
        work_dir_path = Path(work_dir_text)
        output_folder = work_dir_path / "group_data"
        group_size = self.window.spinBoxGroupSize.value()
        if source_mode == "system":
            total_file_count = len(collect_images(source_folder, selected_cams=selected_cams))
        else:
            total_file_count = len(collect_images(source_folder, selected_cams=None))

        result_holder = {"result": None, "error": None}

        def _run_process_data():
            try:
                result_holder["result"] = process_data(
                    source_folder,
                    output_folder,
                    group_size,
                    tiff2png=read_bool(self.window, "checkBoxTiff2Png"),
                    selected_cams=selected_cams,
                    source_mode=source_mode,
                )
            except Exception as exc:
                result_holder["error"] = exc

        existing_file_count = sum(1 for path in output_folder.rglob("*") if path.is_file())
        last_file_count = -1

        def _poll_progress():
            nonlocal last_file_count
            current_file_count = 0
            if output_folder.exists():
                current_file_count = sum(1 for path in output_folder.rglob("*") if path.is_file())
            if current_file_count != last_file_count:
                self.append_log(
                    f"process_data 进度: {current_file_count - existing_file_count} / 总文件数 {total_file_count}"
                )
                last_file_count = current_file_count

        run_thread_with_process_events(_run_process_data, poll_fn=_poll_progress)

        if result_holder["error"] is not None:
            self.append_log(f"数据处理失败: {result_holder['error']}")
            return

        result = result_holder["result"]
        self.append_log(
            "数据处理完成: "
            f"共找到 {result['image_count']} 张图片，生成 {result['group_count']} 个分组，"
            f"输出目录: {result['output_folder']}"
        )
