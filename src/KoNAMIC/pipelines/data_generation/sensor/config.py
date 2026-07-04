from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from KoNAMIC.config.config_utils import require_keys
from KoNAMIC.core.scenarios import (
    ScenarioGenerationConfig,
)


@dataclass(frozen=True)
class SensorSplitGenerationConfig:
    num_traj: int


@dataclass(frozen=True)
class SensorGenerationConfig:
    system_name: str
    modality: str
    controller: str
    seed: int
    scenario_level: str
    dt: float
    t_sim: float

    splits: dict[str, SensorSplitGenerationConfig]

    @property
    def scenario_generation_config(self) -> ScenarioGenerationConfig:
        return ScenarioGenerationConfig(
            dt=self.dt,
            t_sim=self.t_sim,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "SensorGenerationConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, cfg: dict) -> "SensorGenerationConfig":
        require_keys(
            cfg,
            keys=[
                "system_name",
                "modality",
                "controller",
                "seed",
                "simulation",
                "splits",
                "scenario_level",
            ],
            context="sensor generation config",
        )

        simulation = cfg["simulation"]
        require_keys(
            simulation,
            keys=["dt", "t_sim"],
            context="simulation",
        )

        splits = cfg["splits"]

        return cls(
            system_name=cfg["system_name"],
            modality=cfg["modality"],
            controller=str(cfg["controller"]),
            seed=int(cfg["seed"]),
            scenario_level=str(cfg["scenario_level"]),

            dt=float(simulation["dt"]),
            t_sim=float(simulation["t_sim"]),

            splits={
                split_name: SensorSplitGenerationConfig(
                    num_traj=int(split_cfg["num_traj"]),
                )
                for split_name, split_cfg in splits.items()
            },
        )

    def to_dict(self) -> dict:
        data = {
            "system_name": self.system_name,
            "modality": self.modality,
            "controller": self.controller,
            "seed": self.seed,
            "simulation": {
                "dt": self.dt,
                "t_sim": self.t_sim,
            },
            "scenario_level": self.scenario_level,
            "splits": {
                split_name: {
                    "num_traj": split_cfg.num_traj,
                }
                for split_name, split_cfg in self.splits.items()
            },
        }
        return data
