import torch
from torch import nn, Tensor

from KoNAMIC.core.models.model_config import ModelConfig
from .koop_dynamics_standard import KoopDynamicsStandard
from .koop_dynamics_base import KoopDynamicsBase
from .koop_dynamics_structured import KoopDynamicsStructured


class BaseKoopModel(nn.Module):
    """
    Facade class factorizing Koopman latent dynamics (A,B) and common utilities.
    Subclasses must provide the AE I/O (project/reconstruct) and implement forward().

    Rétrocompatibilité:
      - par défaut: old dynamics (same as before)
      - switch via z_dynamics.dynamics_impl = "structured"
    """
    def __init__(self, model_params: ModelConfig):
        super().__init__()
        self.params = model_params
        if model_params.z_dynamics.u_dim is None:
            raise ValueError(
                "model.z_dynamics.u_dim must be set before building a Koopman model. "
                "Call ModelConfig.with_system_dimensions(...) after loading the YAML config."
            )
        self.u_dim = model_params.z_dynamics.u_dim
        self.z_dim = model_params.z_dynamics.z_dim

        impl = self.params.z_dynamics.structured_AB
        if not impl:
            self.dyn: KoopDynamicsBase = KoopDynamicsStandard(model_params)
        elif impl:
            self.dyn = KoopDynamicsStructured(model_params)
        else:
            raise ValueError(f"Unknown dynamics_impl={impl}. Use 'old' or 'structured'.")

        # Keep equilibrium reference if used elsewhere
        # (your training code likely sets self.z_star somewhere)
        self.z_star: Tensor | None = None

    def forward(self, y_init: Tensor, u_traj: Tensor, num_steps: int):
        raise NotImplementedError("Must be implemented by subclass.")

    def forward_controlled(self, x_init: Tensor, u_traj: Tensor, num_steps: int):
        raise NotImplementedError("Must be implemented by subclass.")

    def get_dynamics_parameters(self):
        # si tu as dyn backend (self.dyn)
        if hasattr(self, "dyn"):
            return self.dyn.parameters()
        # ancien fallback
        return list(self.z_drift.parameters()) + list(self.z_act.parameters())

    def construct_koop_matrices(self) -> tuple[Tensor, Tensor]:
        return self.dyn.construct_koop_matrices()

    def z_dynamics_step(self, z_k: Tensor, u_k: Tensor) -> Tensor:
        if self.params.z_dynamics.model == "bilinear":
            if u_k.shape[1] == 2:
                return self.bilinear_dynamics_step_2d(z_k, u_k)
            elif u_k.shape[1] == 4:
                return self.bilinear_dynamics_step_3d(z_k, u_k)
            raise ValueError("Bilinear dynamics only implemented for 2D or 4D inputs.")
        elif self.params.z_dynamics.model == "linear":
            return self.linear_dynamics_step(z_k, u_k)
        raise ValueError("Unknown latent dynamics model")

    def z_dynamics_step_controlled(self, z_k: Tensor, z_k_ref: Tensor):
        raise ValueError("Unknown latent dynamics model")

    def linear_dynamics_step(self, z_k: Tensor, u_k: Tensor) -> Tensor:
        return self.dyn.linear_step(z_k, u_k)

    def bilinear_dynamics_step_2d(self, z_k: Tensor, u_k: Tensor) -> Tensor:
        return self.dyn.bilinear_step_2d(z_k, u_k)

    def bilinear_dynamics_step_3d(self, z_k: Tensor, u_k: Tensor) -> Tensor:
        return self.dyn.bilinear_step_3d(z_k, u_k)

    def equilibrium_loss(self, u_star_scaled: Tensor) -> Tensor:
        # NOTE: identical logic to your current code, only A/B come from backend
        A, B = self.construct_koop_matrices()
        I = torch.eye(self.z_dim, device=A.device, dtype=A.dtype)
        assert self.z_star is not None, "self.z_star must be set before calling equilibrium_loss()."
        z = self.z_star
        r = (A - I) @ z + B @ u_star_scaled
        return (r ** 2).mean()

    # ---------- Debug ----------
    def count_trainable_parameters(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total}\nTrainable parameters: {train}")

    @torch.no_grad()
    def get_bilinear_B_matrices(self) -> torch.Tensor:
        return self.dyn.get_bilinear_B_matrices()

    @torch.no_grad()
    def get_A_matrix(self) -> torch.Tensor:
        return self.dyn.get_A_matrix()

