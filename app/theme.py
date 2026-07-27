from __future__ import annotations

from pathlib import Path

from app.constants import COMBO_ARROW_ICON, COMBO_ARROW_LIGHT_ICON


def load_theme_qss() -> str:
    theme_path = Path(__file__).resolve().parent / "resources" / "theme.qss"
    if theme_path.exists():
        return theme_path.read_text(encoding="utf-8")
    return ""


def apply_theme(window) -> None:
    stylesheet = load_theme_qss()
    if stylesheet:
        window.setStyleSheet(window.styleSheet() + stylesheet)


def apply_checkable_indicator_styles(window) -> None:
    icons_dir = Path(__file__).resolve().parent.parent / "icons"
    checkbox_unchecked = (icons_dir / "checkbox_unchecked.png").as_posix()
    checkbox_checked = (icons_dir / "checkbox_checked.png").as_posix()
    radio_unchecked = (icons_dir / "radio_unchecked.png").as_posix()
    radio_checked = (icons_dir / "radio_checked.png").as_posix()
    window.setStyleSheet(
        window.styleSheet()
        + f"""
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
}}
QCheckBox::indicator:unchecked {{
    image: url({checkbox_unchecked});
}}
QCheckBox::indicator:checked {{
    image: url({checkbox_checked});
}}
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
}}
QRadioButton::indicator:unchecked {{
    image: url({radio_unchecked});
}}
QRadioButton::indicator:checked {{
    image: url({radio_checked});
}}
"""
    )


def work_directory_combo_stylesheet() -> str:
    arrow_icon_path = COMBO_ARROW_ICON.resolve().as_posix()
    return f"""
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


def label_table_combo_stylesheet() -> str:
    arrow_icon_path = COMBO_ARROW_LIGHT_ICON.resolve().as_posix()
    return (
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


def label_table_line_edit_stylesheet() -> str:
    return (
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


def message_box_stylesheet() -> str:
    return (
        "QMessageBox { background-color: #1e293b; color: #f8fafc; }"
        "QMessageBox QLabel { color: #f8fafc; font-size: 13px; }"
        "QPushButton {"
        "  background-color: #2563eb; color: #ffffff;"
        "  border: none; border-radius: 4px;"
        "  padding: 6px 20px; font-size: 13px;"
        "}"
        "QPushButton:hover { background-color: #1d4ed8; }"
    )
