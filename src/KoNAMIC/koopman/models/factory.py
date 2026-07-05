from __future__ import annotations

from typing import TypeAlias

import torch
from sklearn.preprocessing import StandardScaler
from torchinfo import summary

from KoNAMIC import utils, config
from KoNAMIC.koopman.models.model_config import ModelConfig
from KoNAMIC.koopman.lifting.nn.auto_encoder import AutoEncoder
from KoNAMIC.koopman.lifting.nn.auto_encoder_multiview import MultiViewAutoEncoder

from .sensor_koop_model import SensorKoopModel
from .vision_koop_model import VisionKoopModel


VisionModelAndScaler: TypeAlias = tuple[VisionKoopModel, StandardScaler]
SensorModelAndScaler: TypeAlias = tuple[SensorKoopModel, StandardScaler, StandardScaler]
KoopModel: TypeAlias = SensorKoopModel | VisionKoopModel


def build_model(
    modality: config.Modality,
    model_config: ModelConfig,
    device: torch.device | None = None,
) -> KoopModel:
    if modality is config.Modality.SENSOR:
        model, _ = build_sensor_model(model_config, device=device)
        return model
    if modality is config.Modality.VISION:
        model, _ = build_vision_model(model_config, device=device)
        return model
    raise ValueError(f"Unknown modality: {modality}")


def build_vision_model(
    model_config: ModelConfig,
    device: torch.device | None = None,
) -> tuple[VisionKoopModel, torch.device]:
    """
    Build a fresh VisionKoopModel architecture, moved to `device`.
    """
    z_dim = model_config["z_dynamics"]["z_dim"]
    device = utils.load_device() if device is None else device

    auto_encoder_config = model_config["auto_encoder"]
    multi_view = auto_encoder_config["multi_view"]
    if multi_view:
        auto_encoder = MultiViewAutoEncoder(auto_encoder_config, z_dim).to(device)
    else:
        auto_encoder = AutoEncoder(auto_encoder_config, z_dim).to(device)

    model = VisionKoopModel(model_config, auto_encoder).to(device)

    class AEOnly(torch.nn.Module):
        def __init__(self, model: VisionKoopModel) -> None:
            super().__init__()
            self.ae = model.auto_encoder

        def forward(self, y: torch.Tensor) -> torch.Tensor:
            u = torch.zeros(1, 4, device=device)
            z = self.ae.project(y, u)
            return self.ae.reconstruct(z)

    if multi_view:
        summary(AEOnly(model), input_size=(1, 6, 128, 128), depth=5)
    return model, device


def build_sensor_model(
    params: ModelConfig,
    device: torch.device | None = None,
) -> tuple[SensorKoopModel, torch.device]:
    """
    Build a fresh SensorKoopModel architecture, moved to `device`.
    """
    device = utils.load_device() if device is None else device
    model = SensorKoopModel(params).to(device)
    return model, device


# Backward-compatible aliases for checkpoint loading code and older imports.
_build_vision_model = build_vision_model
_build_sensor_model = build_sensor_model
