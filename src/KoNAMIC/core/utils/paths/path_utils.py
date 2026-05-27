import os
import re
import socket
from datetime import datetime
from logging import Logger
from pathlib import Path


def make_timestamp(logger: Logger | None = None) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if logger is not None:
        logger.info(f"Time stamp: {ts}")
    return ts


def make_unique_stamp(base_dir: Path, stamp: str) -> str:
    candidate = stamp
    i = 1

    while (base_dir / candidate).exists():
        candidate = f"{stamp}_{i:02d}"
        i += 1

    return candidate


def make_unique_dir(base_path: Path | str, create: bool = True) -> Path:
    base = Path(base_path)
    path = base
    version = 2

    while path.exists():
        path = base.with_name(f"{base.name}_v{version}")
        version += 1

    if create:
        path.mkdir(parents=True, exist_ok=False)

    return path


def find_project_root(marker: str = "pyproject.toml") -> Path:
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / marker).exists():
            return p
        p = p.parent
    raise FileNotFoundError(f"Cannot find {marker} in parents of {__file__}")


def is_jean_zay_env() -> bool:
    hostname = socket.gethostname()
    fqdn = socket.getfqdn()

    if re.fullmatch(r"jean-zay[1-5]", hostname):
        return True

    if fqdn.endswith("jean-zay.idris.fr"):
        return True

    cluster = os.environ.get("SLURM_CLUSTER_NAME", "").lower()
    return cluster in ("jean-zay", "jeanzay")


def get_project_root() -> Path:
    return find_project_root()


def get_outputs_root() -> Path:
    override = os.environ.get("KSIC_OUTPUTS_ROOT")
    if override:
        return Path(override)

    if is_jean_zay_env():
        return Path(os.environ["SCRATCH"])

    return get_project_root()


def get_datasets_root() -> Path:
    override = os.environ.get("KSIC_DATASETS_ROOT")
    if override:
        return Path(override)

    if is_jean_zay_env():
        return Path(os.environ["SCRATCH"])

    return get_project_root()


def build_base_output_dir(modality: str, run_status: str, drone_dim: int) -> Path:
    return (
        get_outputs_root()
        / "outputs"
        / run_status
        / modality
        / f"{drone_dim}d"
    )


def build_checkpoint_path(run_dir: Path, epoch: int) -> Path:
    return run_dir / "checkpoints" / f"model_epoch_{epoch}.pt"