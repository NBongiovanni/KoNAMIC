#!/usr/bin/env python
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from KoNAMIC.core import utils
from KoNAMIC.core.drone import DroneSpec

from KoNAMIC.pipelines.closed_loop_simulation import (
    run_closed_loop_simulations,
    run_closed_loop_visualization,
    create_reference_builder_factory,
    parse_args_simulation,
    build_default_operating_input,
    build_factory_from_args,
)


def main() -> None:
    args = parse_args_simulation()
    utils.set_seed(args.seed)
    logger = utils.setup_logging()
    project_root = utils.find_project_root()

    drone_config = f"configs/components/drones/{args.drone_dim}d_quadrotor.yaml"
    drone = DroneSpec.from_yaml(project_root / drone_config)

    factory = build_factory_from_args(args=args, logger=logger, drone=drone)

    ctx = factory.build()
    controller = factory.create_controller()
    plant = factory.create_plant()
    simulator = factory.create_simulator(plant, controller)

    reference_builder_factory = create_reference_builder_factory(
        controller_type=args.controller_type,
        modality=args.modality,
        control_params=ctx.control_params,
        drone=drone,
        koop_model=ctx.koop_model,
        x_scaler=ctx.x_scaler,
    )

    u_eq = build_default_operating_input(drone)

    simulation_results = run_closed_loop_simulations(
        num_simulations=ctx.num_simulations,
        control_params=ctx.control_params,
        controller=controller,
        simulator=simulator,
        drone=drone,
        reference_builder_factory=reference_builder_factory,
        u_eq=u_eq,
    )

    run_closed_loop_visualization(
        simulation_indexes=list(range(ctx.num_simulations)),
        drone=drone,
        control_params=ctx.control_params,
        base_ctrl_runs_dir=ctx.control_params["control_runs_dir"],
        simulation_results=simulation_results,
        num_columns=2,
        only_positions=False,
    )


if __name__ == "__main__":
    main()