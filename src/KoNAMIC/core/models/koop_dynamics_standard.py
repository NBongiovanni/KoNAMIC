from __future__ import annotations

import torch
from torch import nn, Tensor

from .koop_dynamics_base import KoopDynamicsBase

class KoopDynamicsStandard(KoopDynamicsBase):
    """
    EXACT copy of your current behavior:
      - A is nn.Linear(z_dim -> z_dim, bias=False)
      - B is nn.Linear(u_dim -> z_dim, bias=False) for linear
      - For bilinear: z_act is nn.Linear(u_dim*z_dyn_dim -> z_dim, bias=False)
    """

    def __init__(self, model_params: dict):
        super().__init__(model_params)
        self.z_drift: nn.Linear | None = None
        self.z_act: nn.Linear | None = None
        self._define_dyn_parameters()

    def _define_dyn_parameters(self) -> None:
        self.z_drift = nn.Linear(self.z_dim, self.z_dim, bias=False)

        if self.params["z_dynamics"]["model"] == "bilinear":
            self.z_act = nn.Linear(self.u_dim * self.z_dyn_dim, self.z_dim, bias=False)
        elif self.params["z_dynamics"]["model"] == "linear":
            self.z_act = nn.Linear(self.u_dim, self.z_dim, bias=False)
        else:
            raise ValueError("Unknown latent dynamics model")

        with torch.no_grad():
            eye = torch.eye(
                self.z_dim,
                dtype=self.z_drift.weight.dtype,
                device=self.z_drift.weight.device
            )
            self.z_drift.weight.copy_(eye)
            self.z_act.weight.zero_()

    def construct_koop_matrices(self) -> tuple[Tensor, Tensor]:
        return self.z_drift.weight, self.z_act.weight

    def linear_step(self, z_k: Tensor, u_k: Tensor) -> Tensor:
        device = z_k.device
        self.z_drift.to(device)
        self.z_act.to(device)
        return self.z_drift(z_k) + self.z_act(u_k)

    def bilinear_step_2d(self, z_k: Tensor, u_k: Tensor) -> Tensor:
        batch_size = z_k.shape[0]
        device = z_k.device
        dtype = z_k.dtype

        assert u_k.shape[1] == 2, "bilinear_dynamics_step suppose u_dim = 2"
        aff = bool(self.params["z_dynamics"]["affine_term"])
        if aff:
            ones = torch.ones(batch_size, 1, device=device, dtype=dtype)
            z_ext = torch.cat([ones, z_k], dim=1)  # [B, z_dyn_dim]
        else:
            z_ext = z_k

        u_k_t = u_k.t()
        z_ext_t = z_ext.t()  # [z_dim+1, B]
        z_u1 = z_ext_t * u_k_t[0].unsqueeze(0)
        z_u2 = z_ext_t * u_k_t[1].unsqueeze(0)
        z_u = torch.cat([z_u1.t(), z_u2.t()], 1)

        act = self.z_act(z_u)
        drift = self.z_drift(z_k)
        return drift + act

    def bilinear_step_3d(self, z_k: Tensor, u_k: Tensor) -> Tensor:
        batch_size = z_k.shape[0]
        device = z_k.device
        dtype = z_k.dtype

        assert u_k.shape[1] == 4, "bilinear_dynamics_step_3d suppose u_dim = 4"

        aff = bool(self.params["z_dynamics"]["affine_term"])
        if aff:
            ones = torch.ones(batch_size, 1, device=device, dtype=dtype)
            z_ext = torch.cat([ones, z_k], dim=1)  # [B, z_dyn_dim]
        else:
            z_ext = z_k

        u_k_t = u_k.t()
        z_ext_t = z_ext.t()  # [z_dim+1, B]
        z_u1 = z_ext_t * u_k_t[0].unsqueeze(0)
        z_u2 = z_ext_t * u_k_t[1].unsqueeze(0)
        z_u3 = z_ext_t * u_k_t[2].unsqueeze(0)
        z_u4 = z_ext_t * u_k_t[3].unsqueeze(0)

        z_u = torch.cat([z_u1.t(), z_u2.t(), z_u3.t(), z_u4.t()], 1)

        act = self.z_act(z_u)
        drift = self.z_drift(z_k)
        return drift + act

    @torch.no_grad()
    def get_bilinear_B_matrices(self) -> torch.Tensor:
        assert self.params["z_dynamics"]["model"] == "bilinear", \
            "Disponible uniquement pour le modèle bilinéaire."

        n = self.z_dim
        m = self.u_dim
        n_dyn = self.z_dyn_dim  # = z_dim + 1

        W = self.z_act.weight  # (n, m*n_dyn)
        B_full = W.view(n, m, n_dyn).permute(1, 0, 2).contiguous()
        return B_full

    @torch.no_grad()
    def get_A_matrix(self) -> torch.Tensor:
        return self.z_drift.weight