import matplotlib
matplotlib.use("Agg")

from KoNAMIC.core import utils
from KoNAMIC.pipelines.open_loop_simulation import (
    OpenLoopMultiVisualizer,
    make_multi_extractors,
    load_simulation_output,
    parse_args_comparison,
    build_comparison_dir,
    build_rollout_results_dir,
    load_open_loop_overlay_preset,
)

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})


def main() -> None:
    args = parse_args_comparison()
    logger = utils.setup_logging()
    cfg = load_open_loop_overlay_preset(args.preset_file, args.preset)
    logger.info("Loaded preset: %s", args.preset)

    cases = utils.load_cases(cfg.modality)

    selected_cases = []
    for model_cfg in cfg.models:
        try:
            case = cases[model_cfg.case_id]
        except (KeyError, IndexError, TypeError):
            raise ValueError(
                f"Invalid case_id={model_cfg.case_id} for modality='{cfg.modality}'."
            )
        selected_cases.append(case)

    if not selected_cases:
        raise ValueError("No models found in configuration.")

    drone_dim = selected_cases[0].drone_dim
    for case in selected_cases[1:]:
        if case.drone_dim != drone_dim:
            raise ValueError("All compared models must share the same drone_dim.")

    names = [m.label for m in cfg.models]
    colors = [m.color for m in cfg.models]
    extractors = make_multi_extractors(cfg.modality)

    comparison_dir = build_comparison_dir(cfg, drone_dim)
    comparison_dir = utils.make_unique_dir(comparison_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Comparison output directory: %s", comparison_dir)
    logger.info("Comparing %d models", len(cfg.models))

    for i in range(cfg.num_traj):
        logger.info("Processing rollout %d / %d", i, cfg.num_traj - 1)

        outputs = []

        for model_cfg, case in zip(cfg.models, selected_cases):
            run_dir = build_rollout_results_dir(
                cfg=cfg,
                drone_dim=drone_dim,
                case=case,
                rollout_idx=i,
            )

            if not run_dir.exists():
                raise FileNotFoundError(
                    f"Missing results directory for model '{model_cfg.label}': {run_dir}"
                )

            logger.info("Loading results for '%s' from %s", model_cfg.label, run_dir)
            outputs.append(load_simulation_output(run_dir))

        traj_out_dir = comparison_dir / f"rollout_{i}"
        traj_out_dir.mkdir(parents=True, exist_ok=True)

        visualizer = OpenLoopMultiVisualizer(
            task=cfg.task,
            dt=args.dt,
            drone_dim=drone_dim,
            only_position=False,
            plot_dir=traj_out_dir,
            names=names,
            num_columns_states=2,
            num_columns_inputs=2,
            colors=colors,
            filename="open_loop_rollout.pdf",
            extractors=extractors,
        )

        visualizer.results_list = outputs
        visualizer.pipeline()

    logger.info("All rollouts processed successfully.")


if __name__ == "__main__":
    main()