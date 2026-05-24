#!/usr/bin/env python
from __future__ import annotations

from logging import Logger
from typing import Optional

from KoNAMIC.core import utils
from KoNAMIC.core.drone import DroneSpec
from KoNAMIC.core.models import load_koop_model_for_eval
from KoNAMIC.core.control.mpc_core import AcadosBackend

from ..simulator import RealControlSimulator
from .prepare_control_params import prepare_control_params
from .factory_context import FactoryContext
from .controller_builder import (
    create_koopman_mpc_controller,
    create_lqr_controller,
    create_pid_controller,
)
from KoNAMIC.pipelines.closed_loop_simulation.factory.plant_builder import create_plant
from KoNAMIC.pipelines.closed_loop_simulation.factory.operating_point import (
    build_default_operating_input,
    get_operating_input_from_config_or_default,
)


class ClosedLoopFactory:
    SUPPORTED_CONTROLLER_TYPES = {
        "koopman_mpc",
        "lqr",
        "pid",
    }

    def __init__(
        self,
        controller_type: str,
        modality: str,
        logger: Logger,
        run_status: str,
        stamp_run: str,
        name_config: str,
        epoch: int,
        geom_losses: bool,
        seed: int,
        drone: DroneSpec,
    ) -> None:
        if controller_type not in self.SUPPORTED_CONTROLLER_TYPES:
            raise ValueError(
                f"Unsupported controller_type='{controller_type}'. "
                f"Supported: {sorted(self.SUPPORTED_CONTROLLER_TYPES)}"
            )

        self.controller_type = controller_type
        self.modality = modality
        self.logger = logger
        self.run_status = run_status
        self.stamp_run = stamp_run
        self.name_config = name_config
        self.epoch = epoch
        self.geom_losses = geom_losses
        self.seed = seed
        self.drone = drone

        self.ctx: Optional[FactoryContext] = None

    def build(self) -> FactoryContext:
        paths = self._build_paths()
        control_params = self._load_control_params(paths)
        num_simulations = int(control_params["num_simulations"])

        ctx = FactoryContext(
            paths=paths,
            control_params=control_params,
            num_simulations=num_simulations,
        )

        if self._requires_koopman_bundle():
            self._attach_koopman_bundle(ctx)

        if self._requires_mpc_backend():
            self._attach_solver_backend(ctx)

        self.ctx = ctx
        return ctx

    def create_controller(self):
        ctx = self._require_ctx()

        if self.controller_type == "koopman_mpc":
            return create_koopman_mpc_controller(self.drone, ctx)

        if self.controller_type == "lqr":
            return create_lqr_controller(self.drone, ctx)

        if self.controller_type == "pid":
            return create_pid_controller(self.drone, ctx)

        raise ValueError(f"Unsupported controller_type='{self.controller_type}'")

    def create_plant(self):
        ctx = self._require_ctx()
        return create_plant(self.drone, self.controller_type, ctx)

    def create_simulator(self, plant, controller):
        ctx = self._require_ctx()
        return RealControlSimulator(ctx.control_params, plant, controller)

    def get_default_operating_input(self):
        return build_default_operating_input(self.drone)

    def get_effective_operating_input(self, control_params: dict):
        return get_operating_input_from_config_or_default(self.drone, control_params)

    def _build_paths(self):
        stamp_control = utils.make_timestamped_dir(self.logger)

        paths = utils.build_run_paths(
            self.modality,
            self.drone.drone_dim,
            self.run_status,
            self.stamp_run,
            None,
            stamp_control,
        )
        paths.closed_loop_eval_dir.mkdir(parents=True, exist_ok=True)
        return paths

    def _load_control_params(self, paths) -> dict:
        if self.controller_type == "koopman_mpc":
            sys_params = utils.load_checkpoint_config(paths)

            control_params = utils.load_base_configs(
                self.name_config,
                "control",
                self.modality,
                self.drone.drone_dim,
                "KNMPC",
            )
            control_params = prepare_control_params(
                sys_params,
                control_params,
                self.epoch,
                paths,
            )
            utils.save_config_yaml(
                control_params,
                paths.closed_loop_eval_dir,
                "control_params.yaml",
            )
            return control_params

        if self.controller_type in {"lqr", "pid"}:
            control_params = utils.load_base_configs(
                self.name_config,
                "control",
                self.modality,
                self.drone.drone_dim,
                self.controller_type,
            )

            utils.save_config_yaml(
                control_params,
                paths.closed_loop_eval_dir,
                "control_params.yaml",
            )
            control_params["control_runs_dir"] = paths.closed_loop_eval_dir
            return control_params

        raise ValueError(f"Unsupported controller_type='{self.controller_type}'")

    def _attach_koopman_bundle(self, ctx: FactoryContext) -> None:
        sys_params = utils.load_checkpoint_config(ctx.paths)
        model_params = sys_params["model_params"]

        koop_model, x_scaler, u_scaler = load_koop_model_for_eval(
            self.modality,
            model_params,
            self.epoch,
            ctx.paths.run_dir,
        )

        ctx.model_params = model_params
        ctx.koop_model = koop_model
        ctx.x_scaler = x_scaler
        ctx.u_scaler = u_scaler

        ctx.control_params = prepare_control_params(
            sys_params,
            ctx.control_params,
            self.epoch,
            ctx.paths,
        )

        utils.save_config_yaml(
            ctx.control_params,
            ctx.paths.closed_loop_eval_dir,
            "control_params.yaml",
        )

    def _attach_solver_backend(self, ctx: FactoryContext) -> None:
        ctx.solver_backend = AcadosBackend(ctx.control_params)

    def _require_ctx(self) -> FactoryContext:
        if self.ctx is None:
            raise RuntimeError("Call build() before creating objects.")
        return self.ctx

    def _requires_koopman_bundle(self) -> bool:
        return self.controller_type == "koopman_mpc"

    def _requires_mpc_backend(self) -> bool:
        return self.controller_type == "koopman_mpc"