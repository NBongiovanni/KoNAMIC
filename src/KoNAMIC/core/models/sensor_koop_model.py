import torch
from torch import Tensor

from KoNAMIC.core.models.model_config import ModelConfig, SensorAutoEncoderConfig
from .base_koop_model import BaseKoopModel
from KoNAMIC.core.models.outputs.sensor_outputs import Pred
from KoNAMIC.core.models.nn.mlp_blocks import build_mlp, build_state_inclusive_mlp, StateSliceDecoder


class SensorKoopModel(BaseKoopModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        if model_config.z_dynamics.x_dim is None:
            raise ValueError("model.z_dynamics.x_dim must be set before building SensorKoopModel.")

        self.x_dim = model_config.z_dynamics.x_dim
        self.u_dim = model_config.z_dynamics.u_dim
        self.z_dim = model_config.z_dynamics.z_dim

        if not isinstance(model_config.auto_encoder, SensorAutoEncoderConfig):
            raise ValueError("SensorKoopModel requires a sensor auto-encoder config.")

        self.auto_encoder_config: SensorAutoEncoderConfig = model_config.auto_encoder
        self.activation = self.auto_encoder_config.activation

        include_state_in_z = self.auto_encoder_config.include_state_in_z
        if include_state_in_z:
            self.encoder = build_state_inclusive_mlp(
                dim_in=self.x_dim,
                latent_extra_dim=self.z_dim - self.x_dim,
                hidden_dims=self.auto_encoder_config.hidden_dims,
                act=self.activation,
            )
            self.decoder = StateSliceDecoder(self.x_dim)
        else:
            self.encoder = build_mlp(
                self.x_dim,
                self.z_dim,
                self.auto_encoder_config.hidden_dims,
                self.activation,
            )
            self.decoder = build_mlp(
                self.z_dim,
                self.x_dim,
                self.auto_encoder_config.hidden_dims,
                self.activation,
            )

    def project(self, y: Tensor) -> Tensor:
        return self.encoder(y)

    def reconstruct(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def batch_projection(self, x_gt: Tensor) -> Tensor:
        b_size = x_gt.size(0)
        n_steps = x_gt.size(1)

        x_gt_flat = torch.reshape(x_gt, (b_size * n_steps, -1))
        z_proj_flat = self.project(x_gt_flat)
        return torch.reshape(z_proj_flat, (b_size, n_steps, self.z_dim))

    def forward(
        self,
        y_init: Tensor,
        u_traj: Tensor,
        num_steps: int,
    ) -> tuple[Tensor, Pred]:

        device = next(self.parameters()).device
        batch_size = u_traj.shape[0]
        dtype = y_init.dtype

        x_pred = torch.zeros(
            (batch_size, num_steps, self.x_dim),
            device=device,
            dtype=dtype
        )
        z_pred = torch.zeros(
            (batch_size, num_steps, self.z_dim),
            device=device,
            dtype=dtype
        )

        z_init_k = self.project(y_init.float())
        x_rec_k = self.reconstruct(z_init_k)

        z_pred[:, 0] = z_init_k
        x_pred[:, 0] = x_rec_k
        x_rec = x_rec_k

        z_pred_k = z_init_k
        for i in range(1, num_steps):
            z_pred_kp1 = self.z_dynamics_step(z_pred_k,u_traj[:, i - 1].to(device))
            z_pred[:, i] = z_pred_kp1
            w_pred_kp1_ = self.reconstruct(z_pred_kp1)
            x_pred[:, i] = w_pred_kp1_
            z_pred_k = z_pred_kp1
        return x_rec, Pred(state=x_pred, z=z_pred)
