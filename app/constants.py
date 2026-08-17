from __future__ import annotations

from enum import Enum
from pathlib import Path

RECENT_WORK_DIRECTORY_KEY = "recentWorkDirectories"
MAX_RECENT_WORK_DIRECTORIES = 10
CONFIG_FILE = Path(__file__).resolve().parent.parent / "VisualFactoryConfig.json"
WORK_DIRECTORY_CONFIG_NAME = "VisualFactoryConfig.json"
CONFIG_VERSION = 1
COMBO_ARROW_ICON = Path(__file__).resolve().parent.parent / "icons" / "combo-arrow-down.svg"
COMBO_ARROW_LIGHT_ICON = Path(__file__).resolve().parent.parent / "icons" / "combo-arrow-down-light.svg"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class LabelUsage(str, Enum):
    UNSET = "请选择用途"
    TRAIN = "用于训练"
    OCCLUDED = "用于遮挡点"


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ModelKind(str, Enum):
    YOLO = "pose"
    OBB = "obb"
    SEG = "seg"
    HRNET = "HRNet"


def current_model_task(window) -> str:
    """根据顶部任务单选按钮返回 'seg' | 'obb' | 'pose'（默认 pose）。"""
    seg = getattr(window, "radioTaskSeg", None)
    if seg is not None and seg.isChecked():
        return "seg"
    obb = getattr(window, "radioTaskObb", None)
    if obb is not None and obb.isChecked():
        return "obb"
    return "pose"
