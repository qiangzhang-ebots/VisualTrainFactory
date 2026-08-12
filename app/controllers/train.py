from __future__ import annotations

import re
import traceback

from app.controllers.base import TabController
from app.widget_io import read_float, read_int, read_text

# 利用数据，开始训练
class TrainController(TabController):
    def _is_obb_task(self) -> bool:
        radio_obb = getattr(self.window, "radioTaskObb", None)
        return bool(radio_obb is not None and radio_obb.isChecked())

    def _get_yolo_model_size(self) -> str:
        """读取用户选择的 YOLO 模型大小 (n/s/m/l/x)，默认 n。"""
        for size in ("n", "s", "m", "l", "x"):
            radio = getattr(self.window, f"radioYoloSize{size.upper()}", None)
            if radio is not None and radio.isChecked():
                return size
        return "n"

    def on_enter(self):
        self.sync_task_ui()

    def sync_task_ui(self):
        is_obb = self._is_obb_task()

        tab_widget = getattr(self.window, "tabWidgetTrainTypes", None)
        hrnet_tab = getattr(self.window, "tabTrainHRNet", None)
        if tab_widget is not None and hrnet_tab is not None:
            index = tab_widget.indexOf(hrnet_tab)
            if index >= 0:
                tab_widget.setTabEnabled(index, not is_obb)
                if is_obb and tab_widget.currentIndex() == index:
                    yolo_tab = getattr(self.window, "tabTrainYolo", None)
                    yolo_index = tab_widget.indexOf(yolo_tab) if yolo_tab is not None else 0
                    tab_widget.setCurrentIndex(max(0, yolo_index))

        btn_hrnet = getattr(self.window, "btnTrainHrnet", None)
        if btn_hrnet is not None:
            btn_hrnet.setEnabled(not is_obb)

        btn_yolo = getattr(self.window, "btnTrainYolo", None)
        if btn_yolo is not None:
            btn_yolo.setText("开始 YOLO OBB 训练" if is_obb else "开始 YOLO 训练")

        weights_edit = getattr(self.window, "lineEditYoloWeights", None)
        if weights_edit is not None:
            current = weights_edit.text().strip()
            model_size = self._get_yolo_model_size()

            # 同步任务类型 (pose/obb)
            if is_obb and current.endswith("-pose.pt"):
                current = current[:-len("-pose.pt")] + "-obb.pt"
            elif not is_obb and current.endswith("-obb.pt"):
                current = current[:-len("-obb.pt")] + "-pose.pt"

            # 同步模型大小 (n/s/m/l/x)
            match = re.match(r"^(yolo26)([nsmlx])(-(?:pose|obb))\.pt$", current)
            if match:
                prefix, _, suffix = match.groups()
                expected = f"{prefix}{model_size}{suffix}.pt"
                if current != expected:
                    current = expected

            weights_edit.setText(current)

    def run_yolo_train(self):
        try:
            from s3_train import trainYolo, trainYoloObb, _get_log_name

            work_dir_text = self.window.comboBoxWorkDirectory.currentText().strip()
            if not work_dir_text:
                self.append_log("请先选择工作目录！")
                return

            label_map = self.window.label_data_preview_controller.get_label_id_mapping()
            if not label_map:
                self.append_log("请先扫描并填写标签ID映射！")
                return

            epochs = read_int(self.window, "spinBoxYoloEpochs", default=100)
            img_size = read_int(self.window, "spinBoxYoloImgSize", default=640)
            batch_size = read_int(self.window, "spinBoxYoloBatch", default=16)
            gpu = read_text(self.window, "lineEditYoloDevice", default="0")
            hflip_ratio = read_float(self.window, "lineEditHflipRatio", default=0.0)
            vflip_ratio = read_float(self.window, "lineEditVflipRatio", default=0.0)
            workers = read_int(self.window, "spinBoxYoloWorkers", default=8)
            weights = read_text(self.window, "lineEditYoloWeights", default="") or None
            model_size = self._get_yolo_model_size()
            log_name = _get_log_name()

            if self._is_obb_task():
                self.append_log(f"开始 YOLO OBB 训练 (模型大小: {model_size})...")
                trainYoloObb(
                    work_dir_text,
                    label_map,
                    epochs,
                    batch_size,
                    img_size,
                    gpu=gpu,
                    logName=log_name,
                    workers=workers,
                    hflipRatio=hflip_ratio,
                    vflipRatio=vflip_ratio,
                    weights=weights,
                    modelSize=model_size,
                )
                self.append_log("YOLO OBB 训练已完成。")
            else:
                self.append_log(f"开始YOLO训练 (模型大小: {model_size})...")
                trainYolo(
                    work_dir_text,
                    label_map,
                    epochs,
                    batch_size,
                    img_size,
                    gpu=gpu,
                    logName=log_name,
                    workers=workers,
                    hflipRatio=hflip_ratio,
                    vflipRatio=vflip_ratio,
                    weights=weights,
                    modelSize=model_size,
                )
                self.append_log("YOLO训练已完成。")
        except Exception as exc:
            self.append_log(f"YOLO训练启动失败: {exc}\n{traceback.format_exc()}")

    def run_hrnet_train(self):
        try:
            if self._is_obb_task():
                self.append_log("当前任务为 OBB，HRNet 不可用。请切换到关键点识别后再训练。")
                return

            from s3_train import trainHRNet, _get_log_name

            work_dir_text = self.window.comboBoxWorkDirectory.currentText().strip()
            if not work_dir_text:
                self.append_log("请先选择工作目录！")
                return

            epochs = read_int(self.window, "spinBoxHrnetEpochs", default=100)
            img_size = read_int(self.window, "lineEditHrnetImgSize", default=640)
            batch_size = read_int(self.window, "spinBoxHrnetBatch", default=16)
            gpu = read_text(self.window, "lineEditHrnetGpu", default="0")

            gpu_ids = [item.strip() for item in str(gpu).split(",") if item.strip()]
            if len(gpu_ids) > 1:
                per_gpu_batch = max(1, (batch_size + len(gpu_ids) - 1) // len(gpu_ids))
                self.append_log(
                    "HRNet 将以多卡分布式模式启动: "
                    f"devices={gpu_ids}, 总 batch={batch_size}, 每卡 batch={per_gpu_batch}"
                )
            else:
                self.append_log(f"HRNet 将以单卡模式启动: device={gpu}, batch={batch_size}")

            log_name = _get_log_name()
            self.append_log("开始HRNet训练...")
            trainHRNet(work_dir_text, epochs, batch_size, img_size, gpu, log_name)
            self.append_log("HRNet训练已完成。")
        except Exception as exc:
            self.append_log(f"HRNet训练启动失败: {exc}\n{traceback.format_exc()}")
