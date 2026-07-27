from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from app.qt_imports import QApplication


def run_thread_with_process_events(worker_fn: Callable[[], None], poll_fn: Optional[Callable[[], None]] = None):
    """Run a background thread while pumping Qt events (preserves original behavior)."""
    worker = threading.Thread(target=worker_fn, daemon=True)
    worker.start()
    while worker.is_alive():
        if poll_fn is not None:
            poll_fn()
        QApplication.processEvents()
        time.sleep(0.2)
    worker.join()
    QApplication.processEvents()
