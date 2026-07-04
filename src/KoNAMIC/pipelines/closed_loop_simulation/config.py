from dataclasses import dataclass
from typing import Any

from KoNAMIC.config.config_utils import require_keys
from KoNAMIC.core.scenarios.config import ScenarioConfig


@dataclass(frozen=True)
class ClosedLoopEvalConfig:
    system_name: str
    modality: str
    enabled: bool
    num_visualized_rollouts: int
    num_rollouts: int
    dt: float
    t_sim: float

    controller_name: str | None = None
    controller_variant: str | None = None
    suppress_output: bool | None = None
    save_plots: bool | None = None
    start_epoch: int | None = None
    dataset_policy: str | None = None
    max_num_trajectories: int | None = None
    batch_size: int | None = None
    num_steps_simulation: int | None = None
    replay_start_epoch: int | None = None

    scenario: ScenarioConfig | None = None
    scenario_level: str | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ClosedLoopEvalConfig":
        require_keys(
            cfg,
            [
                "system_name",
                "modality",
                "enabled",
                "num_visualized_rollouts",
                "num_rollouts",
                "dt",
                "t_sim",
            ],
            "closed_loop_eval",
        )

        scenario = (
            ScenarioConfig.from_dict(cfg["scenario"])
            if "scenario" in cfg and cfg["scenario"] is not None
            else None
        )
        scenario_level = (
            str(cfg["scenario_level"])
            if "scenario_level" in cfg and cfg["scenario_level"] is not None
            else None
        )

        if scenario is None and scenario_level is None:
            raise KeyError(
                "closed_loop_eval requires either 'scenario' or 'scenario_level'."
            )

        return cls(
            system_name=str(cfg["system_name"]),
            modality=str(cfg["modality"]),
            enabled=bool(cfg["enabled"]),
            num_visualized_rollouts=int(cfg["num_visualized_rollouts"]),
            num_rollouts=int(cfg["num_rollouts"]),
            dt=float(cfg["dt"]),
            t_sim=float(cfg["t_sim"]),
            controller_name=(
                str(cfg["controller_name"])
                if cfg.get("controller_name") is not None
                else None
            ),
            controller_variant=(
                str(cfg["controller_variant"])
                if cfg.get("controller_variant") is not None
                else None
            ),
            suppress_output=(
                bool(cfg["suppress_output"])
                if cfg.get("suppress_output") is not None
                else None
            ),
            save_plots=(
                bool(cfg["save_plots"])
                if cfg.get("save_plots") is not None
                else None
            ),
            start_epoch=int(cfg["start_epoch"]) if "start_epoch" in cfg else None,
            dataset_policy=str(cfg["dataset_policy"]) if "dataset_policy" in cfg else None,
            max_num_trajectories=(
                int(cfg["max_num_trajectories"])
                if "max_num_trajectories" in cfg
                else None
            ),
            batch_size=int(cfg["batch_size"]) if "batch_size" in cfg else None,
            num_steps_simulation=(
                int(cfg["num_steps_simulation"])
                if "num_steps_simulation" in cfg
                else None
            ),
            replay_start_epoch=(
                int(cfg["replay_start_epoch"])
                if "replay_start_epoch" in cfg
                else None
            ),
            scenario=scenario,
            scenario_level=scenario_level,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "system_name": self.system_name,
            "modality": self.modality,
            "enabled": self.enabled,
            "num_visualized_rollouts": self.num_visualized_rollouts,
            "num_rollouts": self.num_rollouts,
            "dt": self.dt,
            "t_sim": self.t_sim,
        }

        optional_fields = {
            "controller_name": self.controller_name,
            "controller_variant": self.controller_variant,
            "suppress_output": self.suppress_output,
            "save_plots": self.save_plots,
            "start_epoch": self.start_epoch,
            "dataset_policy": self.dataset_policy,
            "max_num_trajectories": self.max_num_trajectories,
            "batch_size": self.batch_size,
            "num_steps_simulation": self.num_steps_simulation,
            "replay_start_epoch": self.replay_start_epoch,
            "scenario_level": self.scenario_level,
        }
        for key, value in optional_fields.items():
            if value is not None:
                data[key] = value

        if self.scenario is not None:
            data["scenario"] = {
                "profiles": self.scenario.profiles,
                "initial_conditions": self.scenario.initial_conditions,
                "references": self.scenario.references,
            }

        return data
