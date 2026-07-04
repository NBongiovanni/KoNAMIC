import torch
from torch import Tensor

from KoNAMIC.core.rendering.features.geometric_features_diff import compute_centroids_diff, compute_angles_diff
from KoNAMIC.core.models.outputs.vision_outputs import Rec, Pred, ForwardOutputs
from .base_koop_model import BaseKoopModel
from KoNAMIC.core.models.nn.auto_encoder import AutoEncoder


class VisionKoopModel(BaseKoopModel):
    def __init__(self, model_config: dict, auto_encoder: AutoEncoder):
        super().__init__(model_config)
        self.auto_encoder = auto_encoder
        self.num_views = model_config["num_views"]

    # -------- AE I/O (vision) ----------
    def project(self, y: Tensor, u: Tensor) -> Tensor:
        return self.auto_encoder.project(y, u)

    def reconstruct(self, z: Tensor) -> Tensor:
        return self.auto_encoder.reconstruct(z)

    def batch_projection(self, y_traj: Tensor, u_traj) -> Tensor:
        return self.auto_encoder.batch_projection(y_traj, u_traj)

    def forward(
            self,
            y_init: Tensor,
            u_traj: Tensor,
            num_steps: int,
    ) -> ForwardOutputs:

        device = y_init.device
        batch_size = y_init.shape[0]
        H, W = y_init.shape[-2:]
        u_init = u_traj[:, 0]

        y_pred_logits = torch.zeros((batch_size, num_steps - 1, self.num_views, H, W), device=device)
        z_pred = torch.zeros((batch_size, num_steps - 1, self.z_dim), device=device)

        z_k = self.project(y_init, u_init)
        y_rec_init = self.reconstruct(z_k)
        y_rec_init_logits_left = y_rec_init[:, 0:1]
        y_rec_init_left = torch.sigmoid(y_rec_init_logits_left)

        if self.num_views == 2:
            y_rec_init_logits_right = y_rec_init[:, 1:2]
            y_rec_init_right = torch.sigmoid(y_rec_init_logits_right)
        elif self.num_views == 1:
            y_rec_init_logits_right = torch.zeros_like(y_rec_init_logits_left)
            y_rec_init_right = torch.zeros_like(y_rec_init_left)
        else:
            raise ValueError(f"Invalid number of views {self.num_views}")

        for i in range(num_steps - 1):
            z_kp1 = self.z_dynamics_step(z_k, u_traj[:, i])
            y_kp1 = self.reconstruct(z_kp1)

            z_pred[:, i] = z_kp1
            y_pred_logits[:, i] = y_kp1
            z_k = z_kp1

        y_pred_logits_left = y_pred_logits[:, :, 0:1]
        y_pred_left = torch.sigmoid(y_pred_logits_left)
        centroids_pred_left = compute_centroids_diff(y_pred_left)
        angles_pred_left = compute_angles_diff(y_pred_left)

        if self.num_views == 2:
            y_pred_logits_right = y_pred_logits[:, :, 1:2]
            y_pred_right = torch.sigmoid(y_pred_logits_right)
            centroids_pred_right = compute_centroids_diff(y_pred_right)
            angles_pred_right = compute_angles_diff(y_pred_right)
        elif self.num_views == 1:
            y_pred_logits_right = torch.zeros_like(y_pred_logits_left)
            y_pred_right = torch.zeros_like(y_pred_left)
            centroids_pred_right = torch.zeros_like(centroids_pred_left)
            angles_pred_right = torch.zeros_like(angles_pred_left)
        else:
            raise ValueError(f"Invalid number of views {self.num_views}")

        rec = Rec(
            y_logits_right=y_rec_init_logits_right,
            y_right=y_rec_init_right,
            y_logits_left=y_rec_init_logits_left,
            y_left=y_rec_init_left,
        )
        pred = Pred(
            y_left=y_pred_left,
            y_logits_left=y_pred_logits_left,
            y_right=y_pred_right,
            y_logits_right=y_pred_logits_right,
            z=z_pred,
            centroids_left=centroids_pred_left,
            centroids_right=centroids_pred_right,
            angles_left=angles_pred_left,
            angles_right=angles_pred_right
        )
        return ForwardOutputs(rec=rec, pred=pred)