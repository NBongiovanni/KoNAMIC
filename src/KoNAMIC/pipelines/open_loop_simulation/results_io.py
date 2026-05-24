from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import torch


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_simulation_output(run_dir: Path):
    npz_path = run_dir / "results.npz"
    pkl_path = run_dir / "results.pkl"

    if npz_path.exists():
        data = np.load(npz_path)
        return {
            "x_gt": data["x_gt"],
            "x_pred": data["x_pred"],
            "u": data["u"],
        }

    if pkl_path.exists():
        with pkl_path.open("rb") as f:
            outputs = pickle.load(f)

        return {
            "x_gt": _to_numpy(outputs.state_gt_physical),
            "x_pred": _to_numpy(outputs.pred.state),
            "u": _to_numpy(outputs.inputs_physical),
        }

    raise FileNotFoundError(
        f"Missing results file in {run_dir}: expected results.npz or results.pkl"
    )


def convert_old_pkl_to_npz(pkl_path, npz_path):
    with open(pkl_path, "rb") as f:
        outputs = pickle.load(f)

    x_gt = _to_numpy(outputs.state_gt_physical)
    x_pred = _to_numpy(outputs.pred.state)
    u = _to_numpy(outputs.inputs_physical)

    if x_gt is None:
        raise ValueError("state_gt_physical is None in old results.")
    if x_pred is None:
        raise ValueError("outputs.pred.state is None in old results.")
    if u is None:
        raise ValueError("inputs_physical is None in old results.")

    np.savez(
        npz_path,
        x_gt=x_gt,
        x_pred=x_pred,
        u=u,
    )

    print(f"Saved converted results to: {npz_path}")
    print("x_gt:", x_gt.shape)
    print("x_pred:", x_pred.shape)
    print("u:", u.shape)
