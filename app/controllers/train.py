from __future__ import annotations

import re
import traceback

from app.constants import current_model_task
from app.controllers.base import TabController
from app.widget_io import read_float, read_int, read_text

# 利用数据，开始训练
class TrainController(TabController):
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
        task = current_model_task(self.window)
        is_pose = task == "pose"

        tab_widget = getattr(self.window, "tabWidgetTrainTypes", None)
        hrnet_tab = getattr(self.window, "tabTrainHRNet", None)
        if tab_widget is not None and hrnet_tab is not None:
            index = tab_widget.indexOf(hrnet_tab)
            if index >= 0:
                tab_widget.setTabEnabled(index, is_pose)
                if not is_pose and tab_widget.currentIndex() == index:
                    yolo_tab = getattr(self.window, "tabTrainYolo", None)
                    yolo_index = tab_widget.indexOf(yolo_tab) if yolo_tab is not None else 0
                    tab_widget.setCurrentIndex(max(0, yolo_index))

        btn_hrnet = getattr(self.window, "btnTrainHrnet", None)
        if btn_hrnet is not None:
            btn_hrnet.setEnabled(is_pose)

        btn_yolo = getattr(self.window, "btnTrainYolo", None)
        if btn_yolo is not None:
            if task == "seg":
                btn_yolo.setText("开始 YOLO 分割训练")
            elif task == "obb":
                btn_yolo.setText("开始 YOLO OBB 训练")
            else:
                btn_yolo.setText("开始 YOLO 训练")

        weights_edit = getattr(self.window, "lineEditYoloWeights", None)
        if weights_edit is not None:
            current = weights_edit.text().strip()
            model_size = self._get_yolo_model_size()

            # 同步任务类型 (pose/obb/seg)
            suffix_map = {"pose": "-pose.pt", "obb": "-obb.pt", "seg": "-seg.pt"}
            target = suffix_map[task]
            for suffix in suffix_map.values():
                if suffix != target and current.endswith(suffix):
                    current = current[: -len(suffix)] + target
                    break

            # 同步模型大小 (n/s/m/l/x)
            match = re.match(r"^(yolo26)([nsmlx])(-(?:pose|obb|seg))\.pt$", current)
            if match:
                prefix, _, suffix = match.groups()
                expected = f"{prefix}{model_size}{suffix}.pt"
                if current != expected:
                    current = expected

            weights_edit.setText(current)

    def _read_pose_schema_kpt(self):
        """读取 datasets/pose_schema.json 中的关键点数量与名称（新格式）。

        返回 (kptShape, kptNames)；非新格式或无 schema 时返回 (None, None)。
        """
        workspace = self.get_workspace()
        if workspace is None:
            return None, None

        schema_path = workspace.datasets / "pose_schema.json"
        if not schema_path.exists():
            return None, None

        try:
            import json

            with schema_path.open("r", encoding="utf-8") as file:
                schema = json.load(file)
        except (OSError, ValueError):
            return None, None

        if schema.get("pose_format") != "rectangle_point":
            return None, None

        keypoints = schema.get("keypoints", [])
        nfp = schema.get("nfp") or len(keypoints)
        ordered = sorted(keypoints, key=lambda item: item.get("index", 0))
        names = [str(item.get("label", f"p{i + 1}")) for i, item in enumerate(ordered)]
        return [nfp, 3], names

    def run_yolo_train(self):
        try:
            task = current_model_task(self.window)

            from s3_train import trainYolo, trainYoloObb, trainYoloSeg, _get_log_name

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

            # 新格式：从 pose_schema.json 读取真实关键点数量/名称，写入 Pose.yaml
            kpt_shape, kpt_names = self._read_pose_schema_kpt()

            if task == "seg":
                self.append_log(f"开始 YOLO 分割训练 (模型大小: {model_size})...")
                trainYoloSeg(
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
                self.append_log("YOLO 分割训练已完成。")
            elif task == "obb":
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
                    kptShape=kpt_shape,
                    kptNames=kpt_names,
                )
                self.append_log("YOLO训练已完成。")
        except Exception as exc:
            self.append_log(f"YOLO训练启动失败: {exc}\n{traceback.format_exc()}")

    def run_hrnet_train(self):
        try:
            task = current_model_task(self.window)
            if task != "pose":
                self.append_log("当前任务为分割/OBB，HRNet 不可用。请切换到关键点识别后再训练。")
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
