from logging import Logger

from KoNAMIC.core import utils


def generate_run_paths(
    modality: str,
    drone_dim: int,
    dynamics: str,
    run_id: str,
    logger: Logger
) -> utils.RunPaths:

    stamp = utils.make_timestamped_dir(logger)
    if dynamics == "bilinear":
        stamp = "bilin_" + run_id + "_" + stamp
    elif dynamics == "linear":
        stamp = "lin_" + run_id + "_" + stamp
    else:
        raise ValueError("Problem in the dynamics name")

    paths = utils.build_run_paths(
        modality,
        drone_dim,
        "interim",
        stamp,
        stamp_closed_loop="tmp"
    )

    utils.make_unique_dir(paths.run_dir)
    utils.make_unique_dir(paths.log_dir)
    utils.make_unique_dir(paths.closed_loop_eval_dir)
    ckpt_dir = paths.run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return paths