from __future__ import annotations

import torch
from torch import nn, Tensor

from KoNAMIC.koopman.models.model_config import ModelConfig


class KoopDynamicsBase(nn.Module):
    """Interface for latent dynamics implementations."""

    def __init__(self, model_params: ModelConfig):
        super().__init__()
        self.params = model_params
        self.u_dim = model_params.z_dynamics.u_dim
        self.z_dim = model_params.z_dynamics.z_dim

        zdm = self.params.z_dynamics.model
        affine_term = self.params.z_dynamics.affine_term
        if zdm == "bilinear" and affine_term:
            self.z_dyn_dim = self.z_dim + 1
        else:
            self.z_dyn_dim = self.z_dim

    # --- required ---
    def construct_koop_matrices(self) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def get_prunable_params(self):
        raise NotImplementedError

    def linear_step(self, z_k: Tensor, u_k: Tensor) -> Tensor:
        raise NotImplementedError

    def bilinear_step_2d(self, z_k: Tensor, u_k: Tensor) -> Tensor:
        raise NotImplementedError

    def bilinear_step_3d(self, z_k: Tensor, u_k: Tensor) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def get_bilinear_B_matrices(self) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def get_A_matrix(self) -> torch.Tensor:
        raise NotImplementedError
