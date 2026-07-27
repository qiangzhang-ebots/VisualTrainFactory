from __future__ import annotations

from typing import Any, Optional

from app.qt_imports import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox, QWidget


def get_widget(window: QWidget, name: str) -> Optional[QWidget]:
    return getattr(window, name, None)


def require(window: QWidget, name: str) -> QWidget:
    widget = get_widget(window, name)
    if widget is None:
        raise AttributeError(f"Required widget missing: {name}")
    return widget


def read_text(window: QWidget, name: str, default: str = "") -> str:
    widget = get_widget(window, name)
    if widget is None or not isinstance(widget, QLineEdit):
        return default
    text = widget.text().strip()
    return text if text else default


def read_int(window: QWidget, name: str, default: int = 0) -> int:
    widget = get_widget(window, name)
    if widget is None:
        return default
    try:
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        if isinstance(widget, QLineEdit):
            return int(widget.text().strip())
    except (TypeError, ValueError):
        pass
    return default


def read_float(window: QWidget, name: str, default: float = 0.0) -> float:
    widget = get_widget(window, name)
    if widget is None:
        return default
    try:
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, QLineEdit):
            return float(widget.text().strip())
    except (TypeError, ValueError):
        pass
    return default


def read_bool(window: QWidget, name: str, default: bool = False) -> bool:
    widget = get_widget(window, name)
    if widget is None or not isinstance(widget, QCheckBox):
        return default
    return bool(widget.isChecked())


def read_combo_data(window: QWidget, name: str, default: Any = None) -> Any:
    widget = get_widget(window, name)
    if widget is None or not isinstance(widget, QComboBox):
        return default
    try:
        data = widget.currentData()
    except Exception:
        return default
    if isinstance(data, str):
        data = data.strip()
    return data or default
