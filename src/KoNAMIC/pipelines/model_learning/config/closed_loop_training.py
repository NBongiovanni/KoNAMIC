from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from KoNAMIC.config.config_utils import require_keys
from KoNAMIC.core.scenarios.config import ScenarioConfig


@dataclass(frozen=True)
class ClosedLoopTrainingConfig:
    system_name: str
    modality: str
    enabled: bool
    frequency: int
    closed_loop_every: int
    num_rollouts: int
    max_num_trajectories: int
    batch_size: int
    dt: float
    t_sim: float

    controller_name: str | None = None
    controller_variant: str | None = None
    dataset_policy: str | None = None
    num_steps_simulation: int | None = None
    start_epoch: int | None = None
    replay_start_epoch: int | None = None

    scenario: ScenarioConfig | None = None
    scenario_level: str | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ClosedLoopTrainingConfig":
        require_keys(
            cfg,
            [
                "system_name",
                "modality",
                "enabled",
                "frequency",
                "closed_loop_every",
                "num_rollouts",
                "max_num_trajectories",
                "batch_size",
            ],
            "closed_loop_training",
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

        start_epoch = (
            int(cfg["start_epoch"])
            if "start_epoch" in cfg and cfg["start_epoch"] is not None
            else None
        )
        replay_start_epoch = (
            int(cfg["replay_start_epoch"])
            if "replay_start_epoch" in cfg and cfg["replay_start_epoch"] is not None
            else None
        )

        return cls(
            system_name=str(cfg["system_name"]),
            modality=str(cfg["modality"]),
            enabled=bool(cfg["enabled"]),
            frequency=int(cfg["frequency"]),
            closed_loop_every=int(cfg["closed_loop_every"]),
            num_rollouts=int(cfg["num_rollouts"]),
            max_num_trajectories=int(cfg["max_num_trajectories"]),
            batch_size=int(cfg["batch_size"]),
            controller_name=(
                str(cfg["controller_name"])
                if "controller_name" in cfg and cfg["controller_name"] is not None
                else None
            ),
            controller_variant=(
                str(cfg["controller_variant"])
                if "controller_variant" in cfg and cfg["controller_variant"] is not None
                else None
            ),
            dataset_policy=(
                str(cfg["dataset_policy"])
                if "dataset_policy" in cfg and cfg["dataset_policy"] is not None
                else None
            ),
            num_steps_simulation=(
                int(cfg["num_steps_simulation"])
                if "num_steps_simulation" in cfg and cfg["num_steps_simulation"] is not None
                else None
            ),
            dt=(
                float(cfg["dt"])
                if "dt" in cfg and cfg["dt"] is not None
                else None
            ),
            t_sim=(
                float(cfg["t_sim"])
                if "t_sim" in cfg and cfg["t_sim"] is not None
                else None
            ),
            start_epoch=start_epoch,
            replay_start_epoch=replay_start_epoch,
            scenario=scenario,
            scenario_level=scenario_level,
        )

    @property
    def replay_start_epoch_effective(self) -> int | None:
        if self.replay_start_epoch is not None:
            return self.replay_start_epoch
        return self.start_epoch

    def require_scenario_generation_params(self) -> tuple[str, float, float]:
        if self.scenario_level is None:
            raise KeyError(
                "closed_loop_training.scenario_level is required to build the "
                "training scenario generator."
            )
        if self.dt is None or self.t_sim is None:
            raise KeyError(
                "closed_loop_training.dt and closed_loop_training.t_sim are required "
                "to build the training scenario generator."
            )
        return self.scenario_level, self.dt, self.t_sim

    def to_dict(self) -> dict[str, Any]:
        data = {
            "system_name": self.system_name,
            "modality": self.modality,
            "enabled": self.enabled,
            "frequency": self.frequency,
            "closed_loop_every": self.closed_loop_every,
            "num_rollouts": self.num_rollouts,
            "max_num_trajectories": self.max_num_trajectories,
            "batch_size": self.batch_size,
        }

        optional_fields = {
            "controller_name": self.controller_name,
            "controller_variant": self.controller_variant,
            "dataset_policy": self.dataset_policy,
            "num_steps_simulation": self.num_steps_simulation,
            "dt": self.dt,
            "t_sim": self.t_sim,
            "start_epoch": self.start_epoch,
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
