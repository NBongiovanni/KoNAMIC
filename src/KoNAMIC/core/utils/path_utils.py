import os
from pathlib import Path
from typing import Optional
import socket
import re
from dataclasses import dataclass
from datetime import datetime
from logging import Logger


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    open_loop_eval_dir: Optional[Path] = None
    closed_loop_eval_dir: Optional[Path] = None
    log_dir: Optional[Path] = None


def build_plot_path_for_comparison(
        base_run_dir: Path,
        stamp: str,
        stamp_simu: str,
        idx_traj: int
) -> Path:
    return (
        base_run_dir
        / stamp
        / "eval"
        / "open_loop"
        / stamp_simu
        / f"rollout_{idx_traj}"
    )


def build_run_paths(
        modality: str,
        drone_dim: int,
        run_status: str,
        stamp_run: str,
        stamp_open_loop: str | None = None,
        stamp_closed_loop: str | None = None,
) -> RunPaths:
    """Construit les chemins pour un run d'entraînement."""
    runs_base_dir = build_base_output_dir(
        modality,
        run_status,
        drone_dim
    )
    run_dir = runs_base_dir / "models" / stamp_run
    log_dir = runs_base_dir / "logs" / stamp_run

    #TODO: les lignes qui suivent ne sont pas claires
    if stamp_open_loop is not None:
        open_loop_eval_dir = run_dir / "eval" / "open_loop" / stamp_open_loop
    else:
        open_loop_eval_dir = None

    if stamp_closed_loop is not None:
        closed_loop_eval_dir = run_dir / "eval" / "control" / stamp_closed_loop
    else:
        closed_loop_eval_dir = run_dir / "eval" / "control"

    return RunPaths(
        run_dir=run_dir,
        log_dir=log_dir,
        open_loop_eval_dir=open_loop_eval_dir,
        closed_loop_eval_dir=closed_loop_eval_dir,
    )


def build_base_output_dir(
        modality: str, run_status: str, drone_dim: int,
) -> Path:
    jean_zay = is_jean_zay_env()
    root_dir = _get_root_dir_for_runs(jean_zay)
    return root_dir / Path("outputs") / run_status / modality / f"{drone_dim}d"


def build_relative_dataset_path(
        modality: str,
        drone_dim: int,
        version: str,
        phase: str
) -> Path:
    base_dir = Path("datasets")
    return base_dir / modality / f"{drone_dim}d" / version / phase


def build_dataset_path(
        jean_zay: bool,
        drone_dim: int,
        modality: str,
        version: str,
        phase: str
) -> Path:
    relative_dataset_path = build_relative_dataset_path(
        modality,
        drone_dim,
        version,
        phase
    )
    root_dir = _get_root_dir_for_datasets(jean_zay)
    return root_dir / relative_dataset_path


def build_checkpoint_path(path_run_dir: Path, epoch: int) -> Path:
    """
    Build the absolute checkpoint path for a given run directory and epoch.

    Args:
        path_run_dir (Path): Run root directory.
        epoch (int): Checkpoint epoch index.

    Returns:
        Path: Full path to `.../checkpoints/model_epoch_{epoch}.pt`.

    Notes:
        - Uses `find_project_root()` so the path is resolved from the project root.
    """
    root_dir = find_project_root()
    file_model = "model_epoch_{}.pt".format(epoch)

    checkpoint_model_path = Path("checkpoints") / file_model
    return root_dir / path_run_dir / checkpoint_model_path


def make_unique_dir(base_path: Path | str) -> Path:
    base = Path(base_path)
    path = base
    version = 2
    while path.exists():
        path = base.with_name(f"{base.name}_v{version}")
        version += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_timestamped_dir(logger: Logger) -> str:
    """
    Name of the directory where the results will be saved
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"Time stamp :{ts}")
    return ts


def find_project_root(marker: str = "pyproject.toml") -> Path:
    """
    Remonte l’arborescence à partir de ce fichier
    jusqu’à trouver le fichier `marker` (ex. pyproject.toml).
    Author: ChatGPT
    """
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / marker).exists():
            return p
        p = p.parent
    raise FileNotFoundError(f"Impossible de trouver {marker} dans les parents de {__file__}")


def is_jean_zay_env() -> bool:
    hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    # 1) Frontaux : jean-zay1 à jean-zay5
    if re.fullmatch(r"jean-zay[1-5]", hostname):
        return True
    # 2) FQDN se termine par jean-zay.idris.fr
    if fqdn.endswith("jean-zay.idris.fr"):
        return True
    # 3) Slurm : nom du cluster
    cluster = os.environ.get("SLURM_CLUSTER_NAME", "").lower()
    if cluster in ("jean-zay", "jeanzay"):
        return True
    return False


def _get_root_dir_for_runs(jean_zay: bool) -> Path:
    """Détermine le répertoire racine selon l'environnement."""
    if jean_zay:
        return Path(os.environ["SCRATCH"])
    else:
        return find_project_root()


def _get_root_dir_for_datasets(jean_zay: bool) -> Path:
    """
    Racine des data. Priorité à l'override via env var KSIC_DATASETS_ROOT.
    Sinon, on garde l'ancien comportement: SCRATCH sur Jean Zay, project root ailleurs.
    """
    if jean_zay:
        return Path(os.environ["SCRATCH"])
    else:
        return find_project_root()