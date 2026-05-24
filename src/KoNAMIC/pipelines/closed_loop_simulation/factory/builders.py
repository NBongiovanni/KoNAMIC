import argparse
import logging

from KoNAMIC.core.drone import DroneSpec
from KoNAMIC.core.utils.cases_loader import load_cases
from .factory import ClosedLoopFactory


def build_factory_from_args(
    args: argparse.Namespace,
    logger: logging.Logger,
    drone: DroneSpec,
) -> ClosedLoopFactory:
    """
    Build the ClosedLoopFactory from CLI arguments.

    Logic:
    - koopman_mpc: load a predefined case from cases_loader
    - other controllers (e.g. PID): build from direct CLI/config arguments
    """

    if args.controller_type == "koopman_mpc":
        cases = load_cases(args.modality)
        case = cases[args.caseid]
        return build_factory_from_case(case, args, logger, drone)

    return ClosedLoopFactory(
        controller_type=args.controller_type,
        modality=args.modality,
        logger=logger,
        run_status=args.run_status,
        stamp_run=args.controller_type,
        drone=drone,
        name_config=args.config_name,
        epoch=0,
        geom_losses=False,
        seed=args.seed,
    )


def build_factory_from_case(
    case,
    args: argparse.Namespace,
    logger: logging.Logger,
    drone: DroneSpec,
) -> ClosedLoopFactory:
    return ClosedLoopFactory(
        controller_type=args.controller_type,
        modality=args.modality,
        logger=logger,
        run_status=case.run_status,
        stamp_run=case.stamp,
        drone=drone,
        name_config=case.control_config,
        epoch=case.epoch,
        geom_losses=case.geom_losses,
        seed=args.seed,
    )
