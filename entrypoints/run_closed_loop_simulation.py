#!/usr/bin/env python
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from KoNAMIC import paths, config, utils
from KoNAMIC.core.systems import create_system
from KoNAMIC.core.scenarios import build_scenario_generator, load_scenario_gen_config
from KoNAMIC.core.control import (
    build_baseline_controller,
    build_koopman_controller,
    build_default_operating_input,
)
from KoNAMIC.core.simulation import KoopmanClosedLoopSimulator, BaselineClosedLoopSimulator
from KoNAMIC.core.plants import build_plant
from KoNAMIC.core.models import load_koop_model_for_eval
from KoNAMIC.core.models.model_config import ModelConfig
from KoNAMIC.pipelines.closed_loop_simulation import (
    run_closed_loop_simulations,
    run_closed_loop_visualization,
    parse_args_simulation,
    resolve_kmpc_eval_config,
    validate_kmpc_eval_config,
    save_closed_loop_control_config,
)


def main() -> None:
    args = parse_args_simulation()
    modality = config.Modality(args.modality)
    utils.set_seed(args.seed)
    logger = utils.setup_logging()
    system_spec = create_system(args.system_name)

    closed_loop_paths = paths.build_closed_loop_run_paths(
        modality=modality.key,
        system_name=system_spec.system_name,
        stamp_eval=paths.make_timestamp(logger),
        stamp_run=args.stamp_run,
        controller=args.controller,
        run_status=args.run_status,
    )
    closed_loop_paths.eval_dir.mkdir(parents=True, exist_ok=True)

    controller_config = config.load_typed_controller_config_for_system(
        args.controller,
        args.system_name,
        modality.key,
        args.controller_variant,
    )
    eval_config = config.load_typed_closed_loop_eval_config(args.system_name)
    scenario_gen_config = load_scenario_gen_config(
        args.system_name,
        args.scenario_level,
        eval_config.dt,
        eval_config.t_sim,
    )
    plant = build_plant(system_spec, eval_config.dt)
    model_config = None

    if args.controller in {"kmpc", "klqr"}:
        koopman_run_dir = closed_loop_paths.koopman_run_dir
        if koopman_run_dir is None:
            raise RuntimeError(
                "koopman_run_dir must be defined when controller is 'kmpc' or 'klqr'."
            )
        run_config = config.load_yaml(closed_loop_paths.koopman_run_dir / "config.yaml")
        model_raw = run_config["model"] if "model" in run_config else run_config
        model_config = ModelConfig.from_dict(model_raw).with_system_dimensions(
            x_dim=system_spec.x_dim,
            u_dim=system_spec.u_dim,
        )
        if "data_preparation" in run_config:
            delay = run_config["data_preparation"]["postprocessing"]["delay"]
            model_config = model_config.with_delay(delay)
        model_config = model_config.with_consistent_latent_dynamics()

        if args.controller == "kmpc":
            controller_config = resolve_kmpc_eval_config(
                controller_config=controller_config,
                model_config=model_config,
            )
            validate_kmpc_eval_config(
                controller_config=controller_config,
                model_config=model_config,
                controller_variant=args.controller_variant,
            )
        koop_model, data_scalers = load_koop_model_for_eval(
            modality,
            model_config,
            args.epoch,
            closed_loop_paths.koopman_run_dir,
        )

        controller = build_koopman_controller(
            modality,
            data_scalers,
            model_config,
            controller_config,
            closed_loop_paths,
            koop_model,
        )
        simulator = KoopmanClosedLoopSimulator(
            system_spec=system_spec,
            plant=plant,
            controller=controller,
            eval_config=eval_config,
        )
    else:
        controller = build_baseline_controller(system_spec, controller_config)
        simulator = BaselineClosedLoopSimulator(
            system_spec=system_spec,
            plant=plant,
            controller=controller,
            dt=eval_config.dt,
            t_sim=eval_config.t_sim,
        )

    save_closed_loop_control_config(
        eval_dir=closed_loop_paths.eval_dir,
        controller_config=controller_config,
        eval_config=eval_config,
        args=args,
        model_config=model_config,
    )

    scenario_generator = build_scenario_generator(
        system_spec=system_spec,
        cfg=scenario_gen_config,
        seed=args.seed,
    )

    simulation_results = run_closed_loop_simulations(
        num_simulations=eval_config.num_rollouts,
        controller=controller,
        simulator=simulator,
        scenario_generator=scenario_generator,
        u_eq=build_default_operating_input(system_spec),
    )

    run_closed_loop_visualization(
        simulation_indexes=list(range(eval_config.num_rollouts)),
        system_spec=system_spec,
        base_control_runs_dir=closed_loop_paths.eval_dir,
        simulation_results=simulation_results,
        num_columns_states=2,
        num_columns_inputs=2,
        only_positions=False,
    )


if __name__ == "__main__":
    main()
