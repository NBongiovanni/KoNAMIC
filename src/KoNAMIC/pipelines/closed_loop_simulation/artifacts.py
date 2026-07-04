from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from KoNAMIC import config
from KoNAMIC.core.models.model_config import ModelConfig
from KoNAMIC.pipelines.closed_loop_simulation.config import ClosedLoopEvalConfig


def save_closed_loop_control_config(
    *,
    eval_dir: Path,
    controller_config: Any,
    eval_config: ClosedLoopEvalConfig,
    args: Namespace,
    model_config: ModelConfig | None = None,
) -> None:
    """Save the effective closed-loop control configuration next to plots.

    The saved artifact records the controller configuration exactly as loaded
    by the entrypoint, plus the resolved solver options when the controller has
    them. This makes standalone closed-loop plots auditable without going back
    to the training run or component YAML files.
    """

    controller_block = _to_config_dict(controller_config)
    solver_block = _solver_config_block(controller_config)

    payload = {
        "metadata": {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "eval_dir": str(eval_dir),
            "source": "entrypoints/run_closed_loop_simulation.py",
        },
        "cli_args": vars(args),
        "closed_loop_eval": eval_config.to_dict(),
        "controller": controller_block,
    }
    if solver_block is not None:
        payload["solver"] = solver_block
    if model_config is not None:
        payload["model"] = model_config.to_dict()

    config.save_yaml(payload, "control_config.yaml", eval_dir)


def _solver_config_block(controller_config: Any) -> dict[str, Any] | None:
    solver_options = getattr(controller_config, "solver_options", None)
    if solver_options is None:
        return None

    solver_block = {
        "solver_options": _to_config_dict(solver_options),
    }
    solver_options_profile = getattr(controller_config, "solver_options_profile", None)
    if solver_options_profile is not None:
        solver_block["solver_options_profile"] = solver_options_profile
    return solver_block


def _to_config_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj).__name__} does not expose to_dict().")
