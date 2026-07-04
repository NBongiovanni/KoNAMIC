from dataclasses import dataclass
from pathlib import Path

from .paths_utils import get_datasets_root


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    sensor_dir: Path
    vision_dir: Path
    raw_im_dir: Path

    def sensor_split(self, split: str) -> Path:
        return self.sensor_dir / f"{split}.npz"

    @property
    def sensor_generation_config(self) -> Path:
        return self.sensor_dir / "generation_config.yaml"

    def vision_split_dir(self, split: str) -> Path:
        return self.vision_dir / split

    def vision_file(self, split: str)-> Path:
        return self.vision_dir /  split / "dataset_memmap.dat"

    def vision_metadata(self, split: str) -> Path:
        return self.vision_dir / split / "metadata.json"

    @property
    def diagnostics_dir(self) -> Path:
        return self.root / "diagnostics"


def build_dataset_paths(system_name: str, dataset_stamp: str) -> DatasetPaths:
    root = (
        get_datasets_root()
        / "datasets"
        / system_name
        / str(dataset_stamp)
    )

    return DatasetPaths(
        root=root,
        sensor_dir=root / "sensor",
        vision_dir=root / "vision",
        raw_im_dir=root / "raw_images",
    )