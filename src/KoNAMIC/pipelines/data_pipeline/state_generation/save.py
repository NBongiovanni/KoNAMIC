from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Dataset

def save_dataset_npz(dataset: Dataset, metadata: dict, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    print(final_path)

    np.savez_compressed(
        final_path,
        states=dataset.states,
        inputs=dataset.inputs,
        statesRef=dataset.states_ref,
        timeVec=dataset.time,
        metadata=np.array(metadata, dtype=object),
    )