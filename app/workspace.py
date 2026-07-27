from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.constants import ModelKind, Split


@dataclass(frozen=True)
class Workspace:
    root: Path

    @classmethod
    def from_text(cls, work_dir_text: str) -> Optional["Workspace"]:
        text = str(work_dir_text).strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if not path.exists() or not path.is_dir():
            return None
        return cls(path)

    @property
    def group_data(self) -> Path:
        return self.root / "group_data"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    def images_dir(self, split: Split) -> Path:
        return self.datasets / "images" / split.value

    def labels_dir(self, split: Split) -> Path:
        return self.datasets / "labels" / split.value

    def model_runs(self, kind: ModelKind) -> Path:
        return self.runs / kind.value

    def find_label_file(self, stem: str) -> Optional[Path]:
        for split in Split:
            candidate = self.labels_dir(split) / f"{stem}.txt"
            if candidate.exists():
                return candidate
        return None
