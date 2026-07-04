from pathlib import Path

from KoNAMIC import utils, paths, config
from KoNAMIC.core.systems import create_system
from KoNAMIC.core.control.controllers import build_baseline_controller
from KoNAMIC.core.plants import build_plant
from KoNAMIC.core.scenarios import build_scenario_generator, load_scenario_gen_config
from KoNAMIC.pipelines.data_generation import (
    parse_sensor_dataset_generation_args,
    generate_dataset_splits,
    SensorGenerationConfig,
)


def main():
    args = parse_sensor_dataset_generation_args()
    project_root = paths.find_project_root()
    logger = utils.setup_logging()
    system_spec = create_system(args.system_name)

    data_gen_config_path = (
        project_root
        / "configs"
        / "pipelines"
        / "data_generation"
        / args.system_name
        / "sensor.yaml"
    )
    yaml_config = config.load_yaml(data_gen_config_path)
    data_gen_config = SensorGenerationConfig.from_dict(yaml_config)

    scenario_gen_config = load_scenario_gen_config(
        system_spec.system_name,
        data_gen_config.scenario_level,
        data_gen_config.dt,
        data_gen_config.t_sim
    )

    controller_cfg = config.load_typed_controller_config_for_system(
        args.controller,
        args.system_name,
        "sensor",
    )

    dataset_stamp = Path(paths.make_timestamp(logger))
    dataset_paths = paths.build_dataset_paths(args.system_name, str(dataset_stamp))

    plant = build_plant(system_specs=system_spec, dt=data_gen_config.dt)

    controller = build_baseline_controller(system_spec, controller_cfg)
    scenario_generator = build_scenario_generator(
        system_spec=system_spec,
        cfg=scenario_gen_config,
        seed=data_gen_config.seed,
    )

    # ------------------------------------------------------------------
    # Generate all dataset splits (train, val_1, val_2)
    # ------------------------------------------------------------------
    generate_dataset_splits(
        sensor_gen_config=data_gen_config,
        system_spec=system_spec,
        scenario_generator=scenario_generator,
        plant=plant,
        controller=controller,
        output_path=dataset_paths.sensor_dir,
        plot_debug=True,
        seed=args.seed,
    )



if __name__ == "__main__":
    main()
