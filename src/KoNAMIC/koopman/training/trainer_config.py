from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

from KoNAMIC.config.config_utils import require_keys


@dataclass(frozen=True)
class SensorLossWeightsConfig:
    y_pred: float
    z_pred: float
    y_rec: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "SensorLossWeightsConfig":
        require_keys(cfg, ["y_pred", "z_pred", "y_rec"], "trainer.loss_weights")
        return cls(
            y_pred=float(cfg["y_pred"]),
            z_pred=float(cfg["z_pred"]),
            y_rec=float(cfg["y_rec"]),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "y_pred": self.y_pred,
            "z_pred": self.z_pred,
            "y_rec": self.y_rec,
        }


@dataclass(frozen=True)
class VisionLossWeightsConfig:
    y_pred: float
    y_rec: float
    z: float
    c: float
    a: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "VisionLossWeightsConfig":
        require_keys(cfg, ["y_pred", "y_rec", "z", "c", "a"], "trainer.loss_weights")
        return cls(
            y_pred=float(cfg["y_pred"]),
            y_rec=float(cfg["y_rec"]),
            z=float(cfg["z"]),
            c=float(cfg["c"]),
            a=float(cfg["a"]),
        )

    def __getitem__(self, key: str) -> float:
        aliases = {
            "z_pred": "z",
            "centroid": "c",
            "angle": "a",
        }
        return getattr(self, aliases.get(key, key))

    def to_dict(self) -> dict[str, float]:
        return {
            "y_pred": self.y_pred,
            "y_rec": self.y_rec,
            "z": self.z,
            "c": self.c,
            "a": self.a,
        }


LossWeightsConfig: TypeAlias = SensorLossWeightsConfig | VisionLossWeightsConfig


@dataclass(frozen=True)
class SensorOptimizerConfig:
    lr: float
    l2_reg: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "SensorOptimizerConfig":
        require_keys(cfg, ["lr", "l2_reg"], "trainer.optimizer")
        return cls(
            lr=float(cfg["lr"]),
            l2_reg=float(cfg["l2_reg"]),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "lr": self.lr,
            "l2_reg": self.l2_reg,
        }


@dataclass(frozen=True)
class VisionOptimizerLrConfig:
    ae: float
    ab: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "VisionOptimizerLrConfig":
        require_keys(cfg, ["ae", "ab"], "trainer.optimizer.lr")
        return cls(ae=float(cfg["ae"]), ab=float(cfg["ab"]))

    def to_dict(self) -> dict[str, float]:
        return {"ae": self.ae, "ab": self.ab}


@dataclass(frozen=True)
class VisionOptimizerWeightDecayConfig:
    ae: float
    ab: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "VisionOptimizerWeightDecayConfig":
        require_keys(cfg, ["ae", "ab"], "trainer.optimizer.weight_decay")
        return cls(ae=float(cfg["ae"]), ab=float(cfg["ab"]))

    def to_dict(self) -> dict[str, float]:
        return {"ae": self.ae, "ab": self.ab}


@dataclass(frozen=True)
class VisionOptimizerConfig:
    lr: VisionOptimizerLrConfig
    weight_decay: VisionOptimizerWeightDecayConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "VisionOptimizerConfig":
        require_keys(cfg, ["lr", "weight_decay"], "trainer.optimizer")
        return cls(
            lr=VisionOptimizerLrConfig.from_dict(cfg["lr"]),
            weight_decay=VisionOptimizerWeightDecayConfig.from_dict(cfg["weight_decay"]),
        )

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "lr": self.lr.to_dict(),
            "weight_decay": self.weight_decay.to_dict(),
        }


OptimizerConfig: TypeAlias = SensorOptimizerConfig | VisionOptimizerConfig


@dataclass(frozen=True)
class CurriculumConfig:
    phase_epoch_triggers: list[int]
    ramp_duration: int

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CurriculumConfig":
        require_keys(
            cfg,
            ["phase_epoch_triggers", "ramp_duration"],
            "trainer.curriculum",
        )
        return cls(
            phase_epoch_triggers=[int(v) for v in cfg["phase_epoch_triggers"]],
            ramp_duration=int(cfg["ramp_duration"]),
        )

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_epoch_triggers": self.phase_epoch_triggers,
            "ramp_duration": self.ramp_duration,
        }


@dataclass(frozen=True)
class ClosedLoopAugmentationConfig:
    enabled: bool
    frequency: int
    num_rollouts: int
    start_epoch: int
    dataset_policy: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ClosedLoopAugmentationConfig":
        require_keys(
            cfg,
            ["enabled", "frequency", "num_rollouts", "start_epoch", "dataset_policy"],
            "trainer.closed_loop_augmentation",
        )
        return cls(
            enabled=bool(cfg["enabled"]),
            frequency=int(cfg["frequency"]),
            num_rollouts=int(cfg["num_rollouts"]),
            start_epoch=int(cfg["start_epoch"]),
            dataset_policy=str(cfg["dataset_policy"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "frequency": self.frequency,
            "num_rollouts": self.num_rollouts,
            "start_epoch": self.start_epoch,
            "dataset_policy": self.dataset_policy,
        }


@dataclass(frozen=True)
class WandbLoggingConfig:
    project: str
    entity: str | None = None
    mode: str | None = None
    tags: list[str] | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "WandbLoggingConfig":
        require_keys(cfg, ["project"], "trainer.logging.wandb")
        tags = cfg.get("tags")
        return cls(
            project=str(cfg["project"]),
            entity=str(cfg["entity"]) if cfg.get("entity") is not None else None,
            mode=str(cfg["mode"]) if cfg.get("mode") is not None else None,
            tags=[str(tag) for tag in tags] if tags is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {"project": self.project}
        if self.entity is not None:
            data["entity"] = self.entity
        if self.mode is not None:
            data["mode"] = self.mode
        if self.tags is not None:
            data["tags"] = self.tags
        return data


@dataclass(frozen=True)
class LoggingConfig:
    backend: str
    wandb: WandbLoggingConfig | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "LoggingConfig":
        require_keys(cfg, ["backend"], "trainer.logging")
        backend = str(cfg["backend"])
        if backend not in {"tensorboard", "wandb", "both", "none"}:
            raise ValueError(
                "trainer.logging.backend must be one of "
                "tensorboard, wandb, both, or none."
            )

        wandb_cfg = cfg.get("wandb")
        if backend in {"wandb", "both"} and wandb_cfg is None:
            raise KeyError(
                "trainer.logging.wandb is required when backend is "
                f"{backend!r}."
            )

        return cls(
            backend=backend,
            wandb=(
                WandbLoggingConfig.from_dict(wandb_cfg)
                if wandb_cfg is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {"backend": self.backend}
        if self.wandb is not None:
            data["wandb"] = self.wandb.to_dict()
        return data


@dataclass(frozen=True)
class TrainerConfig:
    system_name: str
    modality: str
    seed: int | None
    num_epochs: int
    checkpoint_every: int
    closed_loop_eval_every: int | None
    lr_decay: float
    loss_weights: LossWeightsConfig
    optimizer: OptimizerConfig
    drone_dim: int | None = None
    bilin_reg: bool | None = None
    curriculum: CurriculumConfig | None = None
    closed_loop_augmentation: ClosedLoopAugmentationConfig | None = None
    logging: LoggingConfig | None = None
    grad_clip_max_norm: float | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "TrainerConfig":
        require_keys(
            cfg,
            [
                "system_name",
                "modality",
                "num_epochs",
                "checkpoint_every",
                "loss_weights",
                "optimizer",
                "lr_decay",
            ],
            "trainer",
        )

        modality = str(cfg["modality"])
        if modality == "sensor":
            loss_weights: LossWeightsConfig = SensorLossWeightsConfig.from_dict(cfg["loss_weights"])
            optimizer: OptimizerConfig = SensorOptimizerConfig.from_dict(cfg["optimizer"])
            closed_loop_eval_every = _required_int(cfg, "closed_loop_eval_every", "trainer")
        elif modality == "vision":
            loss_weights = VisionLossWeightsConfig.from_dict(cfg["loss_weights"])
            optimizer = VisionOptimizerConfig.from_dict(cfg["optimizer"])
            closed_loop_eval_every = (
                int(cfg["closed_loop_eval_every"])
                if "closed_loop_eval_every" in cfg and cfg["closed_loop_eval_every"] is not None
                else None
            )
        else:
            raise ValueError(f"Unknown trainer modality: {modality}")

        augmentation_cfg = cfg.get("closed_loop_augmentation")
        logging_cfg = cfg.get("logging")
        curriculum_cfg = cfg.get("curriculum")

        return cls(
            system_name=str(cfg["system_name"]),
            modality=modality,
            seed=int(cfg["seed"]) if "seed" in cfg and cfg["seed"] is not None else None,
            num_epochs=int(cfg["num_epochs"]),
            checkpoint_every=int(cfg["checkpoint_every"]),
            closed_loop_eval_every=closed_loop_eval_every,
            lr_decay=float(cfg["lr_decay"]),
            loss_weights=loss_weights,
            optimizer=optimizer,
            drone_dim=int(cfg["drone_dim"]) if "drone_dim" in cfg and cfg["drone_dim"] is not None else None,
            bilin_reg=bool(cfg["bilin_reg"]) if "bilin_reg" in cfg and cfg["bilin_reg"] is not None else None,
            curriculum=(
                CurriculumConfig.from_dict(curriculum_cfg)
                if curriculum_cfg is not None
                else None
            ),
            closed_loop_augmentation=(
                ClosedLoopAugmentationConfig.from_dict(augmentation_cfg)
                if augmentation_cfg is not None
                else None
            ),
            logging=(
                LoggingConfig.from_dict(logging_cfg)
                if logging_cfg is not None
                else None
            ),
            grad_clip_max_norm=(
                float(cfg["grad_clip_max_norm"])
                if "grad_clip_max_norm" in cfg and cfg["grad_clip_max_norm"] is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "system_name": self.system_name,
            "modality": self.modality,
            "num_epochs": self.num_epochs,
            "checkpoint_every": self.checkpoint_every,
            "lr_decay": self.lr_decay,
            "loss_weights": self.loss_weights.to_dict(),
            "optimizer": self.optimizer.to_dict(),
        }
        optional_fields = {
            "seed": self.seed,
            "closed_loop_eval_every": self.closed_loop_eval_every,
            "drone_dim": self.drone_dim,
            "bilin_reg": self.bilin_reg,
            "curriculum": self.curriculum.to_dict() if self.curriculum is not None else None,
            "grad_clip_max_norm": self.grad_clip_max_norm,
        }
        for key, value in optional_fields.items():
            if value is not None:
                data[key] = value
        if self.closed_loop_augmentation is not None:
            data["closed_loop_augmentation"] = self.closed_loop_augmentation.to_dict()
        if self.logging is not None:
            data["logging"] = self.logging.to_dict()
        return data

    def with_seed(self, seed: int) -> "TrainerConfig":
        return TrainerConfig(
            system_name=self.system_name,
            modality=self.modality,
            seed=seed,
            num_epochs=self.num_epochs,
            checkpoint_every=self.checkpoint_every,
            closed_loop_eval_every=self.closed_loop_eval_every,
            lr_decay=self.lr_decay,
            loss_weights=self.loss_weights,
            optimizer=self.optimizer,
            drone_dim=self.drone_dim,
            bilin_reg=self.bilin_reg,
            curriculum=self.curriculum,
            closed_loop_augmentation=self.closed_loop_augmentation,
            logging=self.logging,
            grad_clip_max_norm=self.grad_clip_max_norm,
        )


def _required_int(cfg: dict[str, Any], key: str, context: str) -> int:
    require_keys(cfg, [key], context)
    return int(cfg[key])
