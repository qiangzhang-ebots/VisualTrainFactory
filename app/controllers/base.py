from __future__ import annotations

from typing import Callable


class TabController:
    def __init__(self, window):
        self.window = window

    def append_log(self, message: str) -> None:
        self.window.append_log(message)

    def get_work_dir_path(self):
        return self.window.get_work_dir_path()

    def get_workspace(self):
        return self.window.get_workspace()
