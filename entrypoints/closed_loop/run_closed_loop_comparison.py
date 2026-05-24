#!/usr/bin/env python
"""
Main script for executing closed-loop control simulations comparison.

Supported modes:
- one model
- one model + PID
- multiple models
- multiple models + PID
"""
import matplotlib
matplotlib.use("Agg")

from KoNAMIC.core import utils
from KoNAMIC.pipelines.closed_loop_simulation import (
    ClosedLoopMultiVisualizer,
    parse_args_comparative_plots,
    load_closed_loop_overlay_preset,
    build_simulation_sources,
)


def main() -> None:
    args = parse_args_comparative_plots()
    cfg = load_closed_loop_overlay_preset(args.preset_file, args.preset)

    logger = utils.setup_logging()
    logger.info("Loaded preset: %s", args.preset)

    sources, drone_dim = build_simulation_sources(cfg)

    comparison_stamp = utils.make_timestamped_dir(logger)
    run_dir = (
        cfg.output_dir
        / cfg.run_status
        / cfg.modality
        / f"{drone_dim}d"
        / "figures"
        / "control"
        / cfg.comparison_name
        / comparison_stamp
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Comparison output directory: %s", run_dir)
    logger.info("Comparing %d sources", len(sources))

    visualizer = ClosedLoopMultiVisualizer(
        drone_dim=drone_dim,
        plot_dir=run_dir,
        dt=cfg.dt,
        names=[src.label for src in sources],
        colors=[src.color for src in sources],
        only_position=False,
        num_columns_states=cfg.num_columns,
        num_columns_inputs=cfg.num_columns,
    )

    result_paths = [src.result_path for src in sources]
    for src in sources:
        if not src.result_path.exists():
            raise FileNotFoundError(
                f"Missing results file for '{src.label}': {src.result_path}"
            )

    visualizer.load_results(*result_paths)
    visualizer.visualize()

    logger.info("Closed-loop comparison finished successfully.")


if __name__ == "__main__":
    main()