from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True)
class DatasetScalers:
    x: Any
    u: Any

    def save(self, run_dir: Path) -> None:
        joblib.dump(self.x, run_dir / "x_scaler.pkl")
        joblib.dump(self.u, run_dir / "u_scaler.pkl")

    @classmethod
    def load(cls, run_dir: Path) -> "DatasetScalers":
        return cls(
            x=joblib.load(run_dir / "x_scaler.pkl"),
            u=joblib.load(run_dir / "u_scaler.pkl"),
        )