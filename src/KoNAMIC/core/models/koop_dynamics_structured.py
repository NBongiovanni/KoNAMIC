from __future__ import annotations

import torch
from torch import nn, Tensor

from .koop_dynamics_base import KoopDynamicsBase

class KoopDynamicsStructured(KoopDynamicsBase):
    """
    "Commande-friendly" structure:
      z = [p; v] with p_{k+1} = p_k + dt v_k
      A = [[I, dt I],
           [A21, A22]]
    And input acts only on v (bottom half):
      Linear:  B = [[0],
                   [B2]]
      Bilinear: the bilinear weights are also zeroed on top half rows.
    """
    def __init__(self, model_params: dict):
        super().__init__(model_params)

        assert self.z_dim % 2 == 0, "KoopDynamicsStructured suppose z_dim pair (split moitié/moitié)."
        self.n1 = self.z_dim // 2
        self.n2 = self.z_dim - self.n1

        dt = float(self.params["dt"])
        self.register_buffer("_I1", torch.eye(self.n1))
        self.register_buffer("_dtI1", dt * torch.eye(self.n1))

        # Learnable blocks
        self.A21 = nn.Parameter(torch.zeros(self.n2, self.n1))
        self.A22 = nn.Parameter(torch.eye(self.n2))

        zdm = self.params["z_dynamics"]["model"]
        if zdm == "linear":
            self.B2 = nn.Parameter(torch.zeros(self.n2, self.u_dim))
        elif zdm == "bilinear":
            # Full bilinear tensor for bottom half only: (n2, u_dim, z_dyn_dim)
            self.B2_full = nn.Parameter(torch.zeros(self.n2, self.u_dim, self.z_dyn_dim))
        else:
            raise ValueError("Unknown latent dynamics model")

    def _assemble_A(self, device, dtype) -> Tensor:
        I1 = self._I1.to(device=device, dtype=dtype)
        dtI1 = self._dtI1.to(device=device, dtype=dtype)

        A = torch.zeros(self.z_dim, self.z_dim, device=device, dtype=dtype)
        A[:self.n1, :self.n1] = I1
        A[:self.n1, self.n1:] = dtI1
        A[self.n1:, :self.n1] = self.A21.to(device=device, dtype=dtype)
        A[self.n1:, self.n1:] = self.A22.to(device=device, dtype=dtype)
        return A

    def _assemble_B_linear(self, device, dtype) -> Tensor:
        B = torch.zeros(self.z_dim, self.u_dim, device=device, dtype=dtype)
        B[self.n1:, :] = self.B2.to(device=device, dtype=dtype)
        return B

    def _assemble_B_full_bilinear(self, device, dtype) -> Tensor:
        # B_full: (u_dim, z_dim, z_dyn_dim), with top half rows forced to 0
        B_full = torch.zeros(self.u_dim, self.z_dim, self.z_dyn_dim, device=device, dtype=dtype)
        # (n2, u_dim, z_dyn_dim) -> (u_dim, n2, z_dyn_dim)
        B_full[:, self.n1:, :] = self.B2_full.to(device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        return B_full

    def construct_koop_matrices(self) -> tuple[Tensor, Tensor]:
        """
        For compatibility with existing save_matrices(), we return:
          - Linear: (A, B) with shapes (z_dim,z_dim) and (z_dim,u_dim)
          - Bilinear: (A, W) where W has shape (z_dim, u_dim*z_dyn_dim), like old z_act.weight
        """
        device = self.A21.device
        dtype = self.A21.dtype
        A = self._assemble_A(device=device, dtype=dtype)

        zdm = self.params["z_dynamics"]["model"]
        if zdm == "linear":
            B = self._assemble_B_linear(device=device, dtype=dtype)
            return A, B

        # bilinear: export W (z_dim, u_dim*z_dyn_dim)
        B_full = self._assemble_B_full_bilinear(device=device, dtype=dtype)  # (u_dim, z_dim, z_dyn_dim)
        W = B_full.permute(1, 0, 2).reshape(self.z_dim, self.u_dim * self.z_dyn_dim).contiguous()
        return A, W

    def linear_step(self, z_k: Tensor, u_k: Tensor) -> Tensor:
        A = self._assemble_A(device=z_k.device, dtype=z_k.dtype)
        B = self._assemble_B_linear(device=z_k.device, dtype=z_k.dtype)
        return z_k @ A.t() + u_k @ B.t()

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

        A = self._assemble_A(device=device, dtype=dtype)
        drift = z_k @ A.t()
        B_full = self._assemble_B_full_bilinear(device=device, dtype=dtype)  # (u_dim, z_dim, z_dyn_dim)

        # sum_i u_i * (z_ext @ B_i^T)
        act = torch.zeros_like(drift)
        for i in range(self.u_dim):
            act = act + u_k[:, i:i+1] * (z_ext @ B_full[i].t())
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

        A = self._assemble_A(device=device, dtype=dtype)
        drift = z_k @ A.t()

        B_full = self._assemble_B_full_bilinear(device=device, dtype=dtype)  # (u_dim, z_dim, z_dyn_dim)

        act = torch.zeros_like(drift)
        for i in range(self.u_dim):
            act = act + u_k[:, i:i+1] * (z_ext @ B_full[i].t())
        return drift + act

    @torch.no_grad()
    def get_bilinear_B_matrices(self) -> torch.Tensor:
        assert self.params["z_dynamics"]["model"] == "bilinear", \
            "Disponible uniquement pour le modèle bilinéaire."
        device = self.A21.device
        dtype = self.A21.dtype
        return self._assemble_B_full_bilinear(device=device, dtype=dtype)

    @torch.no_grad()
    def get_A_matrix(self) -> torch.Tensor:
        device = self.A21.device
        dtype = self.A21.dtype
        return self._assemble_A(device=device, dtype=dtype)