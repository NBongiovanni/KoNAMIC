import contextlib
import os
import pickle
from pathlib import Path


def save_sim_result(result, path: Path) -> None:
    """Sauvegarde via pickle."""
    with open(path, "wb") as f:
        pickle.dump(result, f)


def load_sim_result(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


@contextlib.contextmanager
def suppress_stdout_stderr_fd():
    with open(os.devnull, "w") as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)

        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


@contextlib.contextmanager
def redirect_stdout_stderr_fd(log_path: Path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_file:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)

        try:
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)
            yield log_file
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
