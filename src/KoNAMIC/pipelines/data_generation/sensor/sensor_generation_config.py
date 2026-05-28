from __future__ import annotations
from pathlib import Path
import yaml

from dataclasses import dataclass, field


@dataclass
class SensorSplitGenerationConfig:
    num_traj: int


@dataclass
class SensorGenerationConfig:
    # Global
    seed: int = 0

    # Simulation
    dt: float = 1e-3
    t_sim: float = 3.0

    # Trajectory generation
    x_ref_max: float = 0.5
    y_ref_max: float = 0.5
    z_ref_max: float = 0.5

    x_init_max: float = 0.5
    y_init_max: float = 0.5
    z_init_max: float = 0.5
    angle_init_max: float = 0.0

    tau_ref_min: float = 0.05
    tau_ref_max: float = 0.4

    only_aggressive: bool = False
    init_angles_to_zero: bool = False

    # Controller
    config_controller: str = "configs/control/pid_sensor_2d_base.yaml"

    # Splits
    splits: dict[str, SensorSplitGenerationConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "SensorGenerationConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, cfg: dict) -> "SensorGenerationConfig":
        simulation = cfg.get("simulation", {})
        trajectory_generation = cfg.get("trajectory_generation", {})
        controller = cfg.get("controller", {})
        splits = cfg.get("splits", {})

        reference_bounds = trajectory_generation.get("reference_bounds", {})
        initial_state_bounds = trajectory_generation.get("initial_state_bounds", {})
        reference_time_constants = trajectory_generation.get(
            "reference_time_constants", {}
        )
        options = trajectory_generation.get("options", {})

        return cls(
            seed=cfg.get("seed", 0),

            dt=simulation.get("dt", 1e-3),
            t_sim=simulation.get("t_sim", 3.0),

            x_ref_max=reference_bounds.get("x_max", 0.5),
            y_ref_max=reference_bounds.get("y_max", 0.5),
            z_ref_max=reference_bounds.get("z_max", 0.5),

            x_init_max=initial_state_bounds.get("x_max", 0.5),
            y_init_max=initial_state_bounds.get("y_max", 0.5),
            z_init_max=initial_state_bounds.get("z_max", 0.5),
            angle_init_max=initial_state_bounds.get("angle_max", 0.0),

            tau_ref_min=reference_time_constants.get("tau_min", 0.05),
            tau_ref_max=reference_time_constants.get("tau_max", 0.4),

            only_aggressive=options.get("only_aggressive", False),
            init_angles_to_zero=options.get("init_angles_to_zero", False),

            config_controller=controller.get(
                "config_path",
                "configs/control/pid_sensor_2d_base.yaml",
            ),

            splits={
                split_name: SensorSplitGenerationConfig(
                    num_traj=split_cfg.get("num_traj", 0),
                )
                for split_name, split_cfg in splits.items()
            },
        )