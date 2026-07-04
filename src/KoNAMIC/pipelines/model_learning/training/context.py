from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR, LRScheduler
from torch.utils.tensorboard import SummaryWriter

from KoNAMIC import paths, config
from KoNAMIC.config import Modality
from KoNAMIC.core.models import SensorKoopModel, VisionKoopModel

from .trainer_config import SensorOptimizerConfig, TrainerConfig, VisionOptimizerConfig


@dataclass
class TrainingContext:
    optimizer: Optimizer
    scheduler: LRScheduler
    writer: SummaryWriter


def build_training_context(
    *,
    modality: config.Modality,
    run_paths: paths.RunPaths,
    model: SensorKoopModel | VisionKoopModel,
    trainer_config: TrainerConfig,
) -> TrainingContext:
    if modality is Modality.SENSOR:
        optimizer = _make_sensor_optimizer(trainer_config, model)
    elif modality is Modality.VISION:
        optimizer = _make_vision_optimizer(trainer_config, model)
    else:
        raise ValueError(f"Unknown modality: {modality}")

    scheduler = StepLR(optimizer, step_size=1, gamma=trainer_config.lr_decay)
    writer = SummaryWriter(log_dir=str(run_paths.log_dir))
    model.train()

    return TrainingContext(
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
    )


def _make_sensor_optimizer(
    trainer_config: TrainerConfig,
    model: SensorKoopModel | VisionKoopModel,
) -> Optimizer:
    optimizer_cfg = trainer_config.optimizer
    if not isinstance(optimizer_cfg, SensorOptimizerConfig):
        raise TypeError("Sensor training expects SensorOptimizerConfig.")

    return Adam(
        params=model.parameters(),
        lr=optimizer_cfg.lr,
        weight_decay=optimizer_cfg.l2_reg,
    )


def _make_vision_optimizer(
    trainer_config: TrainerConfig,
    model: SensorKoopModel | VisionKoopModel,
) -> Optimizer:
    optimizer_cfg = trainer_config.optimizer
    if not isinstance(optimizer_cfg, VisionOptimizerConfig):
        raise TypeError("Vision training expects VisionOptimizerConfig.")
    if not isinstance(model, VisionKoopModel):
        raise TypeError("Vision training expects VisionKoopModel.")

    lr_ae = optimizer_cfg.lr.ae
    lr_ab = optimizer_cfg.lr.ab
    wd_ae = optimizer_cfg.weight_decay.ae
    wd_ab = optimizer_cfg.weight_decay.ab

    ae = model.auto_encoder
    params_ae_cnn = list(ae.encoder_cnn.parameters()) + list(ae.decoder_cnn.parameters())
    params_ae_mlp = list(ae.encoder_mlp.parameters()) + list(ae.decoder_mlp.parameters())
    params_ab = _get_dynamics_parameters(model)

    return torch.optim.Adam(
        [
            {"params": params_ae_cnn, "lr": lr_ae, "weight_decay": wd_ae, "name": "ae"},
            {"params": params_ae_mlp, "lr": lr_ae, "weight_decay": wd_ae, "name": "mlp"},
            {"params": params_ab, "lr": lr_ab, "weight_decay": wd_ab, "name": "ab"},
        ]
    )


def _get_dynamics_parameters(model: VisionKoopModel) -> list[torch.nn.Parameter]:
    if hasattr(model, "get_dynamics_parameters"):
        return list(model.get_dynamics_parameters())
    return list(model.z_drift.parameters()) + list(model.z_act.parameters())
