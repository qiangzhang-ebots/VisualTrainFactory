from __future__ import annotations

import traceback

from app.controllers.base import TabController
from app.widget_io import read_float, read_int, read_text

# 利用数据，开始训练
class TrainController(TabController):
    def run_yolo_train(self):
        try:
            from s3_train import trainYolo, _get_log_name

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
            log_name = _get_log_name()

            self.append_log("开始YOLO训练...")
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
            )
            self.append_log("YOLO训练已完成。")
        except Exception as exc:
            self.append_log(f"YOLO训练启动失败: {exc}\n{traceback.format_exc()}")

    def run_hrnet_train(self):
        try:
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
