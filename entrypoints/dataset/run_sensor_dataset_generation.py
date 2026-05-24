import numpy as np
from dataclasses import replace
from pathlib import Path

from KoNAMIC.core import utils
from KoNAMIC.core.drone import build_drone
from KoNAMIC.core.control.controllers import PIDPosAttController
from KoNAMIC.core.plants import Quad3D
from KoNAMIC.pipelines.data_pipeline import DataGenerationConfig, generate_dataset, save_dataset_npz


def main():
    yaml_cfg = utils.load_yaml("../../configs/data/sensor_3d.yaml")
    cfg = DataGenerationConfig(**yaml_cfg)
    ctrl_cfg = utils.load_yaml(yaml_cfg["config_controller"])
    logger = utils.setup_logging()

    drone = build_drone(3)
    plant = Quad3D(1e-3, drone)

    def controller_factory():
        return PIDPosAttController(
            dt=cfg.dt,
            x_dim=12,
            u_dim=4,
            mass=drone.mass,
            inertia=drone.inertia,
            gravity=drone.gravity,
            kp_pos=np.array(ctrl_cfg["kp_pos"]),
            ki_pos=np.array(ctrl_cfg["ki_pos"]),
            kd_pos=np.array(ctrl_cfg["kd_pos"]),
            kp_att=np.array(ctrl_cfg["kp_att"]),
            ki_att=np.array(ctrl_cfg["ki_att"]),
            kd_att=np.array(ctrl_cfg["kd_att"]),
            deriv_filter_n=ctrl_cfg["deriv_filter_N"],
            phi_max=np.deg2rad(ctrl_cfg["phi_ref_max"]),
            theta_max=np.deg2rad(ctrl_cfg["theta_ref_max"]),
            thrust_min=ctrl_cfg["thrust_min"] * drone.mass * drone.gravity,
            thrust_max=ctrl_cfg["thrust_max"] * drone.mass * drone.gravity,
            att_cmd_alpha=ctrl_cfg["att_cmd_alpha"],
            moment_max=np.array(ctrl_cfg["moment_max"]),
            acc_xy_max=ctrl_cfg["acc_xy_max"] * drone.gravity,
            max_moment_rate=np.array(ctrl_cfg["max_moment_rates"]),
        )
    split_idx = 0

    stamp = Path(utils.make_timestamped_dir(logger))
    project_root = utils.find_project_root()
    path = project_root / utils.make_unique_dir("datasets/sensor/3d/" / stamp)

    for split_name, n_traj in cfg.split_lengths.items():
        split_cfg = replace(
            cfg,
            n_traj=n_traj,
            seed=cfg.seed + split_idx,
        )

        dataset, metadata = generate_dataset(
            cfg=split_cfg,
            plant=plant,
            controller_factory=controller_factory,
            split=split_name
        )

        save_dataset_npz(
            dataset,
            metadata,
            path / f"{split_name}_dataset.npz",
        )
        split_idx = split_idx + 1

if __name__ == '__main__':
    main()