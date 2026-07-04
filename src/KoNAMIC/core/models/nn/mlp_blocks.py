from __future__ import annotations

from typing import Callable
import torch
from torch import nn


def _make_activation(act: str) -> Callable[[], nn.Module] | None:
    act = act.lower()
    if act == "relu":
        return lambda: nn.ReLU(inplace=True)
    if act == "tanh":
        return lambda: nn.Tanh()
    if act == "gelu":
        return lambda: nn.GELU()
    if act == "leaky_relu":
        return lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if act in {"identity", "linear", "none"}:
        return None

    raise ValueError(
        f"Unknown activation '{act}'. Supported: relu, tanh, gelu, "
        f"leaky_relu, identity/linear/none, or None"
    )


def build_mlp_old(
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_hidden_layers: int,
        act: str,
) -> nn.Sequential:

    if act == "tanh":
        act_function = nn.Tanh()
    elif act == "relu":
        act_function = nn.ReLU()
    else:
        raise ValueError(f"Unknown activation function {act}")

    layers = []
    layers.append(nn.Linear(dim_in, dim_hidden))
    layers.append(act_function)

    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(dim_hidden, dim_hidden))
        layers.append(act_function)

    layers.append(nn.Linear(dim_hidden, dim_out))
    return nn.Sequential(*layers)


def build_mlp(
        dim_in: int,
        dim_out: int,
        hidden_dims: list[int],
        act: str,
        *,
        dropout: float = 0.0,
        layer_norm: bool = False,
        bias: bool = True,
        control_input: bool = False,
) -> nn.Sequential:
    """
    Build a flexible MLP with arbitrary hidden dimensions.

    Semantics:
      - hidden_dims == []: Linear(dim_in -> dim_out)
      - hidden_dims == [h1, h2, ..., hk]:
            Linear(dim_in -> h1)
            + (act + optional LN + dropout)
            + ...
            Linear(h_{k-1} -> hk)
            + (act + optional LN + dropout)
            + Linear(hk -> dim_out)

    Notes:
      - Fresh activation instance per layer.
      - LayerNorm applied on hidden layers only.
      - Dropout applied after activation.
    """
    act_factory = _make_activation(act)
    layers: list[nn.Module] = []
    dims = [dim_in] + hidden_dims

    # --- Hidden layers ---
    for in_d, out_d in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_d, out_d, bias=bias))
        if layer_norm:
            layers.append(nn.LayerNorm(out_d))
        layers.append(act_factory())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

    # --- Output layer ---
    last_dim = hidden_dims[-1] if hidden_dims else dim_in
    layers.append(nn.Linear(last_dim, dim_out, bias=bias))
    return nn.Sequential(*layers)


class StateSliceDecoder(nn.Module):
    def __init__(self, x_dim: int):
        super().__init__()
        self.x_dim = x_dim

    def forward(self, z):
        return z[..., :self.x_dim]


class StateInclusiveMLP(nn.Module):
    """
    Encoder of the form:
        z = [x ; phi(x)]

    where phi(x) is produced by the original build_mlp.
    Final latent dimension is:
        dim_z = dim_x + latent_extra_dim
    """

    def __init__(
            self,
            dim_in: int,
            latent_extra_dim: int,
            hidden_dims: list[int],
            act: str,
            *,
            dropout: float = 0.0,
            layer_norm: bool = False,
            bias: bool = True,
    ) -> None:
        super().__init__()

        if latent_extra_dim <= 0:
            raise ValueError("latent_extra_dim must be > 0")

        self.dim_in = dim_in
        self.latent_extra_dim = latent_extra_dim
        self.dim_out = dim_in + latent_extra_dim

        # phi(x)
        self.phi = build_mlp(
            dim_in=dim_in,
            dim_out=latent_extra_dim,
            hidden_dims=hidden_dims,
            act=act,
            dropout=dropout,
            layer_norm=layer_norm,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., dim_in)
        returns: (..., dim_in + latent_extra_dim)
        """
        if x.shape[-1] != self.dim_in:
            raise ValueError(
                f"Expected last dimension = {self.dim_in}, got {x.shape[-1]}"
            )

        phi_x = self.phi(x)
        z = torch.cat([x, phi_x], dim=-1)
        return z


def build_state_inclusive_mlp(
        dim_in: int,
        latent_extra_dim: int,
        hidden_dims: list[int],
        act: str,
        *,
        dropout: float = 0.0,
        layer_norm: bool = False,
        bias: bool = True,
) -> StateInclusiveMLP:
    """
    Build an encoder such that:
        z = [x ; phi(x)]

    Args:
        dim_in: dimension of x
        latent_extra_dim: number of additional learned latent coordinates
        hidden_dims: hidden layer dimensions for phi
        act: activation name

    Returns:
        StateInclusiveMLP module
    """
    return StateInclusiveMLP(
        dim_in=dim_in,
        latent_extra_dim=latent_extra_dim,
        hidden_dims=hidden_dims,
        act=act,
        dropout=dropout,
        layer_norm=layer_norm,
        bias=bias,
    )