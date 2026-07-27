"""VisualTrainFactory entry point."""

import multiprocessing
import sys

from app.main_window import VisualTrainFactoryWindow
from app.qt_imports import QApplication


def main():
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = VisualTrainFactoryWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
