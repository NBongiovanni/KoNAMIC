import numpy as np
import torch

from KoNAMIC.core.models import VisionKoopModel
from .plant import Plant

class LearnedModel(Plant):
    def __init__(
            self,
            dt: float,
            koop_model: VisionKoopModel,
    ):
        super().__init__(dt)
        self.dt = dt
        self.koop_model = koop_model
        self.z_dim = koop_model.z_dim
        self.u_dim = self.koop_model.u_dim
        self.discrete_or_continuous = "discrete"
        self.z_dynamics = "linear"

    def _discrete_step(self, z_k: torch.Tensor, u_k: np.ndarray) -> torch.Tensor:
        if self.z_dynamics == "linear":
            a_mat, b_mat = self.koop_model.construct_koop_matrices()
            u_k = torch.tensor(u_k, device=a_mat.device, dtype=a_mat.dtype)
            z_k = torch.tensor(z_k, device=a_mat.device, dtype=a_mat.dtype).squeeze()
            z_next = a_mat @ z_k + b_mat @ u_k
            z_next = z_next.unsqueeze(0)

        elif self.z_dynamics == "bilinear":
            a_mat, act_mat = self.koop_model.construct_koop_matrices()
            b_mat_1 = act_mat[:, :self.z_dim]
            b_mat_2 = act_mat[:, self.z_dim:]
            u_k = torch.tensor(u_k, device=a_mat.device, dtype=a_mat.dtype)
            z_k = torch.tensor(z_k, device=a_mat.device, dtype=a_mat.dtype)

            drift = a_mat @ z_k.t()
            act_1 = (b_mat_1 @ z_k.t()) * u_k[0]
            act_2 = (b_mat_2 @ z_k.t()) * u_k[1]
            act = act_1 + act_2
            z_next = drift.t() + act.t()
        else:
            raise ValueError("Unknown dynamics model")
        return z_next.detach().cpu().numpy()

    def update_image_from_z_k(self, z_k: np.ndarray) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z_k = torch.tensor(z_k).to(device=device)
        im_k = self.koop_model.reconstruct(z_k)
        im_k = torch.sigmoid(im_k)
        im_k = im_k.detach()
        return im_k

    def _dynamics(
            self,
            time,
            z_k: torch.Tensor,
            u_k: np.ndarray
    ) -> torch.Tensor:
        return self._discrete_step(z_k,  u_k)