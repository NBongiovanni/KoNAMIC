from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, TypeAlias

from KoNAMIC.config.config_utils import require_keys


@dataclass(frozen=True)
class ZDynamicsConfig:
    model: str
    z_dim: int
    u_dim: int | None
    structured_AB: bool
    affine_term: bool
    x_dim: int | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ZDynamicsConfig":
        require_keys(
            cfg,
            ["model", "z_dim", "structured_AB", "affine_term"],
            "model.z_dynamics",
        )
        return cls(
            model=str(cfg["model"]),
            z_dim=int(cfg["z_dim"]),
            u_dim=int(cfg["u_dim"]) if "u_dim" in cfg and cfg["u_dim"] is not None else None,
            structured_AB=bool(cfg["structured_AB"]),
            affine_term=bool(cfg["affine_term"]),
            x_dim=int(cfg["x_dim"]) if "x_dim" in cfg and cfg["x_dim"] is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "model": self.model,
            "z_dim": self.z_dim,
            "structured_AB": self.structured_AB,
            "affine_term": self.affine_term,
        }
        if self.u_dim is not None:
            data["u_dim"] = self.u_dim
        if self.x_dim is not None:
            data["x_dim"] = self.x_dim
        return data


@dataclass(frozen=True)
class SensorAutoEncoderConfig:
    include_state_in_z: bool
    hidden_dims: list[int]
    num_hidden_layers: int
    activation: str
    delay: int | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "SensorAutoEncoderConfig":
        require_keys(
            cfg,
            ["include_state_in_z", "hidden_dims", "num_hidden_layers", "activation"],
            "model.auto_encoder",
        )
        return cls(
            include_state_in_z=bool(cfg["include_state_in_z"]),
            hidden_dims=[int(v) for v in cfg["hidden_dims"]],
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            activation=str(cfg["activation"]),
            delay=int(cfg["delay"]) if "delay" in cfg and cfg["delay"] is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "include_state_in_z": self.include_state_in_z,
            "hidden_dims": self.hidden_dims,
            "num_hidden_layers": self.num_hidden_layers,
            "activation": self.activation,
        }
        if self.delay is not None:
            data["delay"] = self.delay
        return data


@dataclass(frozen=True)
class VisionCNNEncoderConfig:
    num_in_channels: int
    out_planes: list[int]
    strides: list[int]
    n_split_layers: int

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "VisionCNNEncoderConfig":
        require_keys(
            cfg,
            ["num_in_channels", "out_planes", "strides", "n_split_layers"],
            "model.auto_encoder.CNN.encoder",
        )
        return cls(
            num_in_channels=int(cfg["num_in_channels"]),
            out_planes=[int(v) for v in cfg["out_planes"]],
            strides=[int(v) for v in cfg["strides"]],
            n_split_layers=int(cfg["n_split_layers"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_in_channels": self.num_in_channels,
            "out_planes": self.out_planes,
            "strides": self.strides,
            "n_split_layers": self.n_split_layers,
        }


@dataclass(frozen=True)
class VisionCNNDecoderConfig:
    num_out_channels: int
    out_planes: list[int]
    strides: list[int]

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "VisionCNNDecoderConfig":
        require_keys(
            cfg,
            ["num_out_channels", "out_planes", "strides"],
            "model.auto_encoder.CNN.decoder",
        )
        return cls(
            num_out_channels=int(cfg["num_out_channels"]),
            out_planes=[int(v) for v in cfg["out_planes"]],
            strides=[int(v) for v in cfg["strides"]],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_out_channels": self.num_out_channels,
            "out_planes": self.out_planes,
            "strides": self.strides,
        }


@dataclass(frozen=True)
class VisionCNNConfig:
    encoder: VisionCNNEncoderConfig
    decoder: VisionCNNDecoderConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "VisionCNNConfig":
        require_keys(cfg, ["encoder", "decoder"], "model.auto_encoder.CNN")
        return cls(
            encoder=VisionCNNEncoderConfig.from_dict(cfg["encoder"]),
            decoder=VisionCNNDecoderConfig.from_dict(cfg["decoder"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder.to_dict(),
            "decoder": self.decoder.to_dict(),
        }


@dataclass(frozen=True)
class VisionMLPConfig:
    num_hidden_layers: int
    hidden_dims: list[int]
    layer_norm: bool | None = None
    dropout: float | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "VisionMLPConfig":
        require_keys(
            cfg,
            ["num_hidden_layers", "hidden_dims"],
            "model.auto_encoder.MLP",
        )
        return cls(
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            hidden_dims=[int(v) for v in cfg["hidden_dims"]],
            layer_norm=bool(cfg["layer_norm"]) if "layer_norm" in cfg and cfg["layer_norm"] is not None else None,
            dropout=float(cfg["dropout"]) if "dropout" in cfg and cfg["dropout"] is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_dims": self.hidden_dims,
        }
        if self.layer_norm is not None:
            data["layer_norm"] = self.layer_norm
        if self.dropout is not None:
            data["dropout"] = self.dropout
        return data


@dataclass(frozen=True)
class VisionAutoEncoderConfig:
    multi_view: bool
    control_inputs: bool
    H: int
    W: int
    CNN: VisionCNNConfig
    MLP: VisionMLPConfig
    activation: str
    include_state_in_z: bool | None = None
    split_indices: list[list[int]] | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "VisionAutoEncoderConfig":
        require_keys(
            cfg,
            ["multi_view", "control_inputs", "H", "W", "CNN", "MLP", "activation"],
            "model.auto_encoder",
        )
        split_indices = None
        if "split_indices" in cfg and cfg["split_indices"] is not None:
            split_indices = [[int(v) for v in values] for values in cfg["split_indices"]]

        return cls(
            multi_view=bool(cfg["multi_view"]),
            control_inputs=bool(cfg["control_inputs"]),
            H=int(cfg["H"]),
            W=int(cfg["W"]),
            CNN=VisionCNNConfig.from_dict(cfg["CNN"]),
            MLP=VisionMLPConfig.from_dict(cfg["MLP"]),
            activation=str(cfg["activation"]),
            include_state_in_z=(
                bool(cfg["include_state_in_z"])
                if "include_state_in_z" in cfg and cfg["include_state_in_z"] is not None
                else None
            ),
            split_indices=split_indices,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "multi_view": self.multi_view,
            "control_inputs": self.control_inputs,
            "H": self.H,
            "W": self.W,
            "CNN": self.CNN.to_dict(),
            "MLP": self.MLP.to_dict(),
            "activation": self.activation,
        }
        if self.include_state_in_z is not None:
            data["include_state_in_z"] = self.include_state_in_z
        if self.split_indices is not None:
            data["split_indices"] = self.split_indices
        return data


AutoEncoderConfig: TypeAlias = SensorAutoEncoderConfig | VisionAutoEncoderConfig


@dataclass(frozen=True)
class ModelConfig:
    system_name: str
    modality: str
    dt: float
    z_dynamics: ZDynamicsConfig
    auto_encoder: AutoEncoderConfig
    drone_dim: int | None = None
    num_views: int | None = None
    delay: int | None = None
    state_dim: int | None = None
    input_dim: int | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ModelConfig":
        require_keys(
            cfg,
            ["system_name", "modality", "z_dynamics", "dt", "auto_encoder"],
            "model",
        )
        modality = str(cfg["modality"])
        if modality == "sensor":
            auto_encoder = SensorAutoEncoderConfig.from_dict(cfg["auto_encoder"])
        elif modality == "vision":
            auto_encoder = VisionAutoEncoderConfig.from_dict(cfg["auto_encoder"])
        else:
            raise ValueError(f"Unknown model modality: {modality}")

        return cls(
            system_name=str(cfg["system_name"]),
            modality=modality,
            dt=float(cfg["dt"]),
            z_dynamics=ZDynamicsConfig.from_dict(cfg["z_dynamics"]),
            auto_encoder=auto_encoder,
            drone_dim=int(cfg["drone_dim"]) if "drone_dim" in cfg and cfg["drone_dim"] is not None else None,
            num_views=int(cfg["num_views"]) if "num_views" in cfg and cfg["num_views"] is not None else None,
            delay=int(cfg["delay"]) if "delay" in cfg and cfg["delay"] is not None else None,
            state_dim=int(cfg["state_dim"]) if "state_dim" in cfg and cfg["state_dim"] is not None else None,
            input_dim=int(cfg["input_dim"]) if "input_dim" in cfg and cfg["input_dim"] is not None else None,
            notes=str(cfg["notes"]) if "notes" in cfg and cfg["notes"] is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "system_name": self.system_name,
            "modality": self.modality,
            "dt": self.dt,
            "z_dynamics": self.z_dynamics.to_dict(),
            "auto_encoder": self.auto_encoder.to_dict(),
        }
        optional_fields = {
            "drone_dim": self.drone_dim,
            "num_views": self.num_views,
            "delay": self.delay,
            "state_dim": self.state_dim,
            "input_dim": self.input_dim,
            "notes": self.notes,
        }
        for key, value in optional_fields.items():
            if value is not None:
                data[key] = value
        return data

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def with_system_dimensions(self, x_dim: int, u_dim: int) -> "ModelConfig":
        return replace(
            self,
            z_dynamics=replace(self.z_dynamics, x_dim=x_dim, u_dim=u_dim),
            state_dim=x_dim,
            input_dim=u_dim,
        )

    def with_delay(self, delay: int) -> "ModelConfig":
        if isinstance(self.auto_encoder, SensorAutoEncoderConfig):
            return replace(
                self,
                auto_encoder=replace(self.auto_encoder, delay=delay),
            )
        return replace(self, delay=delay)

    def with_latent_dynamics(self, latent_dynamics: str) -> "ModelConfig":
        z_dynamics = replace(self.z_dynamics, model=latent_dynamics)
        if latent_dynamics == "linear":
            z_dynamics = replace(z_dynamics, affine_term=False)
        return replace(self, z_dynamics=z_dynamics)

    def with_consistent_latent_dynamics(self) -> "ModelConfig":
        if self.z_dynamics.model == "linear" and self.z_dynamics.affine_term:
            return replace(
                self,
                z_dynamics=replace(self.z_dynamics, affine_term=False),
            )
        return self

    def with_state_in_z(self, include_state_in_z: bool) -> "ModelConfig":
        if not isinstance(self.auto_encoder, SensorAutoEncoderConfig):
            raise ValueError("with_state_in_z is only valid for sensor model configs.")
        auto_encoder = replace(
            self.auto_encoder,
            include_state_in_z=include_state_in_z,
        )
        z_dynamics = self.z_dynamics
        if include_state_in_z:
            z_dynamics = replace(z_dynamics, structured_AB=False)
        return replace(self, auto_encoder=auto_encoder, z_dynamics=z_dynamics)
