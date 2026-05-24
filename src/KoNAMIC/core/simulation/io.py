from pathlib import Path
import pickle


from .trajectories import ClosedLoopTrajectory

def save_sim_result(result: ClosedLoopTrajectory, path: Path) -> None:
    """Sauvegarde via pickle."""
    with open(path, "wb") as f:
        pickle.dump(result, f)
