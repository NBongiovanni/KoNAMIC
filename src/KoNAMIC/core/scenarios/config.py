from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from KoNAMIC import config, paths


@dataclass(frozen=True)
class ScenarioConfig:
    profiles: dict[str, float]
    initial_conditions: dict[str, Any]
    references: dict[str, Any]

    @classmethod
    def from_dict(cls, cfg: dict) -> "ScenarioConfig":
        config.require_keys(
            cfg,
            keys=[
                "profiles",
                "initial_conditions",
                "references",
            ],
            context="scenario",
        )

        return cls(
            profiles=dict(cfg["profiles"]),
            initial_conditions=dict(cfg["initial_conditions"]),
            references=dict(cfg["references"]),
        )


@dataclass(frozen=True)
class ScenarioGenerationConfig:
    dt: float
    t_sim: float
    scenario: ScenarioConfig



def load_scenario_gen_config(
        system_name: str, scenario_level: str, dt: float, t_sim: float
) -> ScenarioGenerationConfig:
    project_root = paths.find_project_root()
    config_path = (
            project_root
            / "configs"
            / "scenarios"
            / f"{system_name}"
            / f"{scenario_level}.yaml"
    )
    cfg = config.load_yaml(config_path)
    scenario_config = ScenarioConfig.from_dict(cfg)
    scenario_gen_config = ScenarioGenerationConfig(dt=dt, t_sim=t_sim, scenario=scenario_config)
    return scenario_gen_config
