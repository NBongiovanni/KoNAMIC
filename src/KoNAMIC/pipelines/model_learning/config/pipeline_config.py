from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from KoNAMIC import config, paths
from KoNAMIC.core.control.config import (
    KlqrControllerConfig,
    KmpcControllerConfig,
    load_controller_config,
    load_controller_config_from_dict,
)
from KoNAMIC.koopman.models.model_config import ModelConfig
from KoNAMIC.koopman.training.trainer_config import TrainerConfig
from KoNAMIC.pipelines.closed_loop_simulation.config import ClosedLoopEvalConfig
from KoNAMIC.pipelines.open_loop_simulation.config import OpenLoopEvalConfig
from KoNAMIC.pipelines.data_preparation.sensor.config import SensorPreparationConfig
from KoNAMIC.pipelines.data_preparation.vision.config import VisionPreparationConfig
from .closed_loop_training import ClosedLoopTrainingConfig
from .open_loop_training import OpenLoopTrainingConfig


@dataclass
class TrainingPipelineConfig:
    model: ModelConfig
    trainer: TrainerConfig
    data_preparation: SensorPreparationConfig | VisionPreparationConfig
    controller: KmpcControllerConfig | KlqrControllerConfig
    closed_loop_eval_controller: KmpcControllerConfig | KlqrControllerConfig
    closed_loop_training_controller: KmpcControllerConfig | KlqrControllerConfig
    closed_loop_training: ClosedLoopTrainingConfig
    open_loop_training: OpenLoopTrainingConfig
    closed_loop_eval: ClosedLoopEvalConfig
    open_loop_eval: OpenLoopEvalConfig

    @property
    def prediction_horizon(self) -> PredictionHorizon:
        return PredictionHorizon.from_data_preparation(self.data_preparation)

    @classmethod
    def from_config_blocks(
        cls,
        model_config: dict,
        trainer_config: dict,
        controller_config: dict,
        data_preparation_config: dict,
        open_loop_training_config: dict,
        closed_loop_training_config: dict,
        open_loop_eval_config: dict,
        closed_loop_eval_config: dict,
    ) -> "TrainingPipelineConfig":
        trainer = TrainerConfig.from_dict(trainer_config)
        if trainer.modality == "sensor":
            data_preparation = SensorPreparationConfig.from_dict(data_preparation_config)
        elif trainer.modality == "vision":
            data_preparation = VisionPreparationConfig.from_dict(data_preparation_config)
        else:
            raise ValueError(f"Unknown trainer modality: {trainer.modality}")

        controller = _load_koopman_training_controller_config(controller_config)
        closed_loop_training = ClosedLoopTrainingConfig.from_dict(closed_loop_training_config)
        closed_loop_eval = ClosedLoopEvalConfig.from_dict(closed_loop_eval_config)

        return cls(
            model=ModelConfig.from_dict(model_config),
            trainer=trainer,
            controller=controller,
            closed_loop_eval_controller=_resolve_closed_loop_controller_config(
                default_controller=controller,
                system_name=closed_loop_eval.system_name,
                modality=closed_loop_eval.modality,
                controller_name=closed_loop_eval.controller_name,
                controller_variant=closed_loop_eval.controller_variant,
                config_section="closed_loop_eval",
            ),
            closed_loop_training_controller=_resolve_closed_loop_controller_config(
                default_controller=controller,
                system_name=closed_loop_training.system_name,
                modality=closed_loop_training.modality,
                controller_name=closed_loop_training.controller_name,
                controller_variant=closed_loop_training.controller_variant,
                config_section="closed_loop_training",
            ),
            data_preparation=data_preparation,
            open_loop_training=OpenLoopTrainingConfig.from_dict(open_loop_training_config),
            closed_loop_training=closed_loop_training,
            open_loop_eval=OpenLoopEvalConfig.from_dict(open_loop_eval_config),
            closed_loop_eval=closed_loop_eval,
        )

    @classmethod
    def load_default(
        cls,
        system_name: str,
        modality: config.Modality,
        controller_name: str = "kmpc",
        controller_variant: str | None = None,
    ) -> "TrainingPipelineConfig":
        root = paths.find_project_root()
        config_dir = root / "configs"

        config_paths = cls._build_config_paths(
            config_dir=config_dir,
            system_name=system_name,
            modality=modality.key,
            controller_name=controller_name,
            controller_variant=controller_variant,
        )

        configs = {
            name: config.load_yaml(path)
            for name, path in config_paths.items()
        }

        cfg = cls.from_config_blocks(**configs)

        return cfg

    @staticmethod
    def _build_config_paths(
        config_dir: Path,
        system_name: str,
        modality: str,
        controller_name: str = "kmpc",
        controller_variant: str | None = None,
    ) -> dict[str, Path]:
        controller_config_path = (
            config_dir
            / "components"
            / "controllers"
            / controller_name
            / system_name
            / modality
            / f"{controller_variant}.yaml"
            if controller_variant is not None
            else config_dir
            / "components"
            / "controllers"
            / controller_name
            / system_name
            / f"{modality}.yaml"
        )

        return {
            "trainer_config": (
                config_dir
                / "pipelines"
                / "training"
                / "trainer"
                / system_name
                / f"{modality}.yaml"
            ),
            "model_config": (
                config_dir
                / "components"
                / "models"
                / system_name
                / f"{modality}.yaml"
            ),
            "controller_config": controller_config_path,
            "data_preparation_config": (
                config_dir
                / "pipelines"
                / "data_preparation"
                / system_name
                / f"{modality}.yaml"
            ),
            "open_loop_training_config": (
                config_dir
                / "pipelines"
                / "training"
                / "open_loop"
                / system_name
                / f"{modality}.yaml"
            ),
            "closed_loop_training_config": (
                config_dir
                / "pipelines"
                / "training"
                / "closed_loop"
                / system_name
                / f"{modality}.yaml"
            ),
            "open_loop_eval_config": (
                config_dir
                / "pipelines"
                / "evaluation"
                / "open_loop"
                / system_name
                / f"{modality}.yaml"
            ),
            "closed_loop_eval_config": (
                config_dir
                / "pipelines"
                / "evaluation"
                / "closed_loop"
                / system_name
                / f"{modality}.yaml"
            ),
        }

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "trainer": self.trainer.to_dict(),
            "data_preparation": self.data_preparation.to_dict(),
            "controller": self.controller.to_dict(),
            "closed_loop_eval_controller": self.closed_loop_eval_controller.to_dict(),
            "closed_loop_training_controller": self.closed_loop_training_controller.to_dict(),
            "open_loop_training": self.open_loop_training.to_dict(),
            "closed_loop_training": self.closed_loop_training.to_dict(),
            "open_loop_eval": self.open_loop_eval.to_dict(),
            "closed_loop_eval": self.closed_loop_eval.to_dict(),
        }

    def sync_shared_params(self, system) -> None:
        self.model = self.model.with_system_dimensions(
            x_dim=system.x_dim,
            u_dim=system.u_dim,
        )
        self.model = self.model.with_delay(self.data_preparation.postprocessing.delay)

        self.controller = _with_controller_dt(self.controller, self.model.dt)
        self.closed_loop_eval_controller = _with_controller_dt(
            self.closed_loop_eval_controller,
            self.model.dt,
        )
        self.closed_loop_training_controller = _with_controller_dt(
            self.closed_loop_training_controller,
            self.model.dt,
        )

    def apply_cli_options(self, args: Namespace) -> None:
        self.model = self.model.with_latent_dynamics(args.latent_dynamics)

        if args.modality == "sensor":
            self.model = self.model.with_state_in_z(args.state_in_z)

        self.trainer = self.trainer.with_seed(args.seed)

    def resolve_derived_params(self) -> None:
        self.model = _resolve_model_derived_params(self.model)

        solver_profile = _solver_options_profile_for_latent_dynamics(
            self.model.z_dynamics.model,
        )
        self.controller = _with_kmpc_solver_options_profile(
            self.controller,
            solver_profile,
        )
        self.closed_loop_eval_controller = _with_kmpc_solver_options_profile(
            self.closed_loop_eval_controller,
            solver_profile,
        )
        self.closed_loop_training_controller = _with_kmpc_solver_options_profile(
            self.closed_loop_training_controller,
            solver_profile,
        )


def _resolve_model_derived_params(model_config: ModelConfig) -> ModelConfig:
    if (
        model_config.modality == "sensor"
        and hasattr(model_config.auto_encoder, "include_state_in_z")
        and model_config.auto_encoder.include_state_in_z
    ):
        return model_config.with_state_in_z(True)
    return model_config


def _resolve_closed_loop_controller_config(
    *,
    default_controller: KmpcControllerConfig | KlqrControllerConfig,
    system_name: str,
    modality: str,
    controller_name: str | None,
    controller_variant: str | None,
    config_section: str,
) -> KmpcControllerConfig | KlqrControllerConfig:
    if controller_name is None and controller_variant is None:
        return default_controller

    controller = controller_name or default_controller.controller_type
    loaded_config = load_controller_config(
        controller=controller,
        system_name=system_name,
        modality=modality,
        variant=controller_variant,
    )
    if isinstance(loaded_config, (KmpcControllerConfig, KlqrControllerConfig)):
        return loaded_config

    raise TypeError(
        f"{config_section} supports Koopman closed-loop controllers only. "
        f"Got {type(loaded_config).__name__}."
    )


def _with_kmpc_solver_options_profile(
    controller_config: KmpcControllerConfig | KlqrControllerConfig,
    profile: str,
) -> KmpcControllerConfig | KlqrControllerConfig:
    if isinstance(controller_config, KmpcControllerConfig):
        return controller_config.with_solver_options_profile(profile)
    return controller_config


def _solver_options_profile_for_latent_dynamics(latent_dynamics: str) -> str:
    if latent_dynamics == "linear":
        return "linear_latent_medium"
    if latent_dynamics == "bilinear":
        return "bilinear_latent_medium"
    raise ValueError(
        "Cannot derive KMPC solver_options_profile from "
        f"model.z_dynamics.model={latent_dynamics!r}."
    )


def _with_controller_dt(
    controller_config: KmpcControllerConfig | KlqrControllerConfig,
    dt: float,
) -> KmpcControllerConfig | KlqrControllerConfig:
    updated = controller_config.with_dt(dt)
    if isinstance(updated, (KmpcControllerConfig, KlqrControllerConfig)):
        return updated
    raise TypeError(f"Unexpected controller config after with_dt: {type(updated).__name__}.")


def _load_koopman_training_controller_config(
    controller_config: dict,
) -> KmpcControllerConfig | KlqrControllerConfig:
    loaded_config = load_controller_config_from_dict(controller_config)
    if isinstance(loaded_config, (KmpcControllerConfig, KlqrControllerConfig)):
        return loaded_config

    raise TypeError(
        "TrainingPipelineConfig supports Koopman closed-loop controllers only. "
        f"Got {type(loaded_config).__name__}."
    )


@dataclass(frozen=True)
class PredictionHorizon:
    train: int
    val: list[int]

    @classmethod
    def from_dataset_params(cls, dataset_params: dict) -> "PredictionHorizon":
        return cls(
            train=dataset_params["train"]["num_steps_pred"],
            val=[
                val_dataset["num_steps_pred"]
                for val_dataset in dataset_params["val_datasets"]
            ],
        )

    @classmethod
    def from_data_preparation(
            cls,
            dataset_params: SensorPreparationConfig | VisionPreparationConfig,
    ) -> "PredictionHorizon":
        return cls(
            train=dataset_params.train.num_steps_pred,
            val=[val.num_steps_pred for val in dataset_params.val_datasets],
        )