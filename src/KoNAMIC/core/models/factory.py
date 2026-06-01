from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from KoNAMIC.core import utils
from KoNAMIC.core.models.nn.auto_encoder import AutoEncoder
from KoNAMIC.core.models.nn.auto_encoder_multiview import MultiViewAutoEncoder

from .sensor_koop_model import SensorKoopModel
from .vision_koop_model import VisionKoopModel


@dataclass
class TrainingContext:
    optimizer: Optimizer
    scheduler: _LRScheduler
    writer: SummaryWriter


VisionModelAndTC: TypeAlias = tuple[VisionKoopModel, TrainingContext]
SensorModelAndTC: TypeAlias = tuple[SensorKoopModel, TrainingContext]
VisionModelAndScaler: TypeAlias = tuple[VisionKoopModel, StandardScaler]
SensorModelAndScaler: TypeAlias = tuple[SensorKoopModel, StandardScaler, StandardScaler]


def init_model(modality: str, run_paths, model_params: dict, training_params: dict):
    if modality == "sensor":
        return init_sensor_model(run_paths, model_params, training_params)
    elif modality == "vision":
        return init_vision_model(model_params, training_params)
    else:
        raise ValueError("Unknown modality")


def init_vision_model(model_params: dict, training_params: dict):
    """
    Initialize a fresh Koopman model and its training context.

    Args:
        model_params: Model hyperparameters.
        training_params: Training hyperparameters.

    Returns:
        (model, training_context)
    """
    model, device = _build_vision_model(model_params)
    optimizer = _make_optimizer_vision(training_params["optimizer"], model)
    scheduler = StepLR(optimizer, step_size=1, gamma=training_params["lr_decay"])
    writer = SummaryWriter(log_dir=str(training_params["log_dir"]))

    model.train()

    training_context = TrainingContext(
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
    )
    return model, training_context


def init_sensor_model(
        run_paths: utils.RunPaths, model_params: dict, training_params: dict
):
    model, device = _build_sensor_model(model_params)

    optimizer_cfg = training_params["optimizer"]
    optimizer = Adam(
        params=model.parameters(),
        lr=optimizer_cfg["lr"],
        weight_decay=optimizer_cfg["l2_reg"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=1,
        gamma=training_params["lr_decay"],
    )
    writer = SummaryWriter(log_dir=str(run_paths.log_dir))

    training_context = TrainingContext(
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
    )
    return model, training_context


def _build_vision_model(
        model_params: dict, device: torch.device | None = None,
) -> tuple[VisionKoopModel, torch.device]:
    """
    Build a fresh VisionKoopModel (architecture only), moved to `device`.

    This is meant to be used by BOTH:
      - init_vision_koop_model (fresh training)
      - load_vision_koop_model_for_train/eval (restore from checkpoint)

    Returns:
        (model, device)
    """
    z_dim = model_params["z_dynamics"]["z_dim"]
    device = utils.load_device() if device is None else device

    multi_view = model_params["auto_encoder"]["multi_view"]
    if multi_view:
        auto_encoder = MultiViewAutoEncoder(model_params["auto_encoder"], z_dim).to(device)
    else:
        auto_encoder = AutoEncoder(model_params["auto_encoder"], z_dim).to(device)

    model = VisionKoopModel(model_params, auto_encoder).to(device)

    class AEOnly(torch.nn.Module):
        def __init__(self, model: VisionKoopModel) -> None:
            super().__init__()
            self.ae = model.auto_encoder

        def forward(self, y: torch.Tensor) -> torch.Tensor:
            u = torch.zeros(1, 4, device=device)
            z = self.ae.project(y, u)
            y_rec = self.ae.reconstruct(z)
            return y_rec
    if multi_view:
        summary(AEOnly(model), input_size=(1, 6, 128, 128), depth=5)
    return model, device


def _build_sensor_model(
    params: dict,
    device: torch.device | None = None,
) -> tuple[SensorKoopModel, torch.device]:
    """
    Build a fresh SensorKoopModel (architecture only), moved to `device`.

    This is meant to be used by BOTH:
      - init_sensor_koop_model
      - load_sensor_koop_model_for_eval (and possible future load_for_train)

    Returns:
        (model, device)
    """
    device = utils.load_device() if device is None else device
    model = SensorKoopModel(params).to(device)
    return model, device


def _get_dynamics_parameters(model: VisionKoopModel) -> list[torch.nn.Parameter]:
    """
    Backward-compatible helper:
    - new API: model.get_dynamics_parameters()
    - old API: model.z_drift + model.z_act
    """
    if hasattr(model, "get_dynamics_parameters"):
        return list(model.get_dynamics_parameters())
    return list(model.z_drift.parameters()) + list(model.z_act.parameters())


def _make_optimizer_vision(hparams: dict, model: VisionKoopModel) -> Optimizer:
    lr_ae = hparams["lr"]["ae"]
    lr_ab = hparams["lr"]["ab"]
    wd_ae = hparams["weight_decay"]["ae"]
    wd_ab = hparams["weight_decay"]["ab"]

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