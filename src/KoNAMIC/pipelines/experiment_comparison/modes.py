from __future__ import annotations

from argparse import Namespace
from logging import Logger

from KoNAMIC import config, paths, utils
from KoNAMIC.core.systems import create_system
from KoNAMIC.paths.open_loop_paths import build_comparison_dir
from KoNAMIC.pipelines.closed_loop_simulation import (
    ClosedLoopMultiVisualizer,
    closed_loop_trajectory_to_comparison_result,
)
from KoNAMIC.pipelines.open_loop_simulation import (
    OpenLoopMultiVisualizer,
    load_open_loop_comparison_result,
)

from .sources import (
    ComparisonSource,
    build_closed_loop_sources,
    build_open_loop_sources,
)
from .context import build_experiment_comparison_context
from .result_indices import resolve_common_result_indices
from .runner import run_indexed_comparison
from .trajectories import TrajectoryComparisonResult


def run_open_loop_comparison(args: Namespace, logger: Logger) -> None:
    context = build_experiment_comparison_context(
        args,
        config.load_open_loop_overlay_preset,
    )
    cfg = context.cfg
    system_spec = create_system(args.system_name)
    logger.info("Loaded open-loop preset: %s", args.preset)

    comparison_dir = build_comparison_dir(cfg, context.system_dim)
    comparison_dir = paths.make_unique_dir(comparison_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Comparison output directory: %s", comparison_dir)
    logger.info("Comparing %d experiments", len(cfg.models))

    sources = build_open_loop_sources(
        cfg=cfg,
        cases=context.cases,
        system_dim=context.system_dim,
    )

    rollout_indices = resolve_common_result_indices(
        sources=sources,
        directory_prefix="rollout_",
        result_filename="results.npz",
        source_kind_label="model",
        empty_intersection_message=(
            "No common open-loop rollout indices found across compared experiments."
        ),
    )
    logger.info("Rendering %d open-loop rollouts", len(rollout_indices))

    def load_result(_source, result_path):
        return load_open_loop_comparison_result(result_path.parent)

    def make_visualizer(_index, plot_dir, results):
        dt = _resolve_rollout_dt(outputs=list(results))
        return OpenLoopMultiVisualizer(
            task=cfg.task,
            dt=dt,
            system_spec=system_spec,
            only_position=False,
            plot_dir=plot_dir,
            names=[source.label for source in sources],
            num_columns_states=2,
            num_columns_inputs=2,
            colors=[source.color for source in sources],
            filename="open_loop_rollout.pdf",
        )

    run_indexed_comparison(
        sources=sources,
        indices=rollout_indices,
        output_dir=comparison_dir,
        directory_prefix="rollout_",
        result_filename="results.npz",
        item_label="open-loop rollout",
        logger=logger,
        load_result=load_result,
        make_visualizer=make_visualizer,
    )

    logger.info("Open-loop comparison finished successfully.")


def run_closed_loop_comparison(args: Namespace, logger: Logger) -> None:
    context = build_experiment_comparison_context(
        args,
        config.load_closed_loop_overlay_preset,
    )
    cfg = context.cfg
    system_spec = create_system(args.system_name)
    logger.info("Loaded closed-loop preset: %s", args.preset)

    sources, _system_dim = build_closed_loop_sources(
        cfg=cfg,
        cases=context.cases,
    )
    run_status = _resolve_comparison_run_status(cfg, sources)

    comparison_dir = (
        cfg.output_dir
        / run_status
        / cfg.modality
        / cfg.system_name
        / "figures"
        / "closed_loop"
        / cfg.comparison_name
    )
    comparison_dir = paths.make_unique_dir(comparison_dir)

    logger.info("Comparison output directory: %s", comparison_dir)
    logger.info("Comparing %d sources", len(sources))

    run_indices = resolve_common_result_indices(
        sources=sources,
        directory_prefix="run_",
        result_filename="results.pkl",
        source_kind_label="source",
        empty_intersection_message=(
            "No common closed-loop run indices found across compared sources."
        ),
    )
    logger.info("Rendering %d closed-loop trajectories", len(run_indices))

    def load_result(_source, result_path):
        return closed_loop_trajectory_to_comparison_result(
            utils.load_sim_result(result_path),
            dt=cfg.dt,
        )

    def make_visualizer(_index, plot_dir, _results):
        return ClosedLoopMultiVisualizer(
            system_spec=system_spec,
            plot_dir=plot_dir,
            dt=cfg.dt,
            names=[source.label for source in sources],
            colors=[source.color for source in sources],
            only_position=False,
            num_columns_states=cfg.num_columns,
            num_columns_inputs=cfg.num_columns,
            filename="closed_loop_simulation.pdf",
        )

    run_indexed_comparison(
        sources=sources,
        indices=run_indices,
        output_dir=comparison_dir,
        directory_prefix="run_",
        result_filename="results.pkl",
        item_label="closed-loop trajectory",
        logger=logger,
        load_result=load_result,
        make_visualizer=make_visualizer,
    )

    logger.info("Closed-loop comparison finished successfully.")


def _resolve_rollout_dt(
    *,
    outputs: list[TrajectoryComparisonResult],
) -> float:
    dts = [out.dt for out in outputs]

    reference_dt = dts[0]
    mismatched = [dt for dt in dts[1:] if abs(dt - reference_dt) > 1e-12]
    if mismatched:
        raise ValueError(
            f"Compared rollouts have inconsistent dt values: {[reference_dt, *mismatched]}"
        )

    return reference_dt


def _resolve_comparison_run_status(cfg, sources: list[ComparisonSource]) -> str:
    if cfg.run_status is not None:
        return cfg.run_status

    source_run_statuses = {src.run_status for src in sources}
    if len(source_run_statuses) == 1:
        return source_run_statuses.pop()

    raise ValueError(
        "Compared sources use multiple run_status values. "
        "Set 'run_status' explicitly in the comparison preset."
    )
