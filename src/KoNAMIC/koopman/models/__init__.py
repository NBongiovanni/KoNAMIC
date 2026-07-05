from .checkpoints import (
    load_vision_model_for_eval,
    load_sensor_koop_model_for_eval,
    load_koop_model_for_eval,
)
from importlib import import_module
import sys

from .factory import build_model
from .vision_koop_model import VisionKoopModel
from .sensor_koop_model import SensorKoopModel
from .base_koop_model import BaseKoopModel
from .model_config import (
    AutoEncoderConfig,
    ModelConfig,
    SensorAutoEncoderConfig,
    VisionAutoEncoderConfig,
    VisionCNNConfig,
    VisionCNNDecoderConfig,
    VisionCNNEncoderConfig,
    VisionMLPConfig,
    ZDynamicsConfig,
)
from KoNAMIC.koopman.models.outputs.sensor_outputs import SensorValForwardOutputs
from KoNAMIC.koopman.models.outputs.vision_outputs import (
    ForwardOutputs,
    VisionValForwardOutputs,
    Rec,
    Pred,
    GroundTruth
)

_LIFTING_ALIASES = (
    "nn",
    "nn.auto_encoder",
    "nn.auto_encoder_multiview",
    "nn.conv_decoder",
    "nn.conv_encoder",
    "nn.layers",
    "nn.mlp_blocks",
    "nn.multi_view_decoder",
    "nn.multi_view_encoder",
)

for _name in _LIFTING_ALIASES:
    sys.modules[f"{__name__}.{_name}"] = import_module(f"KoNAMIC.koopman.lifting.{_name}")

del import_module, sys, _LIFTING_ALIASES, _name
