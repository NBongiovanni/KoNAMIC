import torch
from torch import nn, Tensor

from KoNAMIC.core.models.nn.conv_encoder import ConvEncoder
from KoNAMIC.core.models.nn.conv_decoder import ConvDecoder
from KoNAMIC.core.models.nn.mlp_blocks import build_mlp


class AutoEncoder(nn.Module):
    def __init__(self, params: dict, z_dim: int):
        super().__init__()
        self.params = params
        self.z_dim = z_dim
        self.control_inputs = self.params["control_inputs"]

        self.encoder_cnn = ConvEncoder(
            num_in_channels=self.params["CNN"]["encoder"]["num_in_channels"],
            H=self.params["H"],
            W=self.params["W"],
            out_planes=self.params["CNN"]["encoder"]["out_planes"],
            strides=self.params["CNN"]["encoder"]["strides"],
            n_split_layers = self.params["CNN"]["encoder"]["n_split_layers"],
        )
        self.w_dim = self.encoder_cnn.latent_dim

        self.encoder_mlp = build_mlp(
            self.w_dim,
            self.z_dim,
            self.params["MLP"]["hidden_dims"],
            self.params["activation"],
            layer_norm=self.params["MLP"].get("layer_norm", False),
            dropout=self.params["MLP"].get("dropout", 0.0),
            control_input=self.params["control_inputs"],
        )

        self.decoder_mlp = build_mlp(
            self.z_dim,
            self.w_dim,
            self.params["MLP"]["hidden_dims"],
            self.params["activation"],
            layer_norm=self.params["MLP"].get("layer_norm", False),
            dropout=self.params["MLP"].get("dropout", 0.0),
        )

        self.decoder_cnn = ConvDecoder(
            out_channels=self.params["CNN"]["decoder"]["num_out_channels"],
            feature_shape=self.encoder_cnn.feature_shape,
            out_planes=self.params["CNN"]["decoder"]["out_planes"],
            strides=self.params["CNN"]["decoder"]["strides"],
        )

    def project(self, y: Tensor, u: Tensor) -> Tensor:
        """
        Projection: image space to the observable space
        """
        w = self.encoder_cnn(y)
        if self.control_inputs:
            return self.encoder_mlp(w, u)
        else:
            return self.encoder_mlp(w)

    def reconstruct(self, z: Tensor) -> Tensor:
        """
        Reconstruction: observable space to the image space
        """
        w = self.decoder_mlp(z)
        return self.decoder_cnn(w)

    def forward(self, y: Tensor, u: Tensor | None = None) -> Tensor:
        z = self.project(y, u)
        return self.reconstruct(z)

    def batch_projection(
            self,
            y_gt: Tensor,
            u_traj: Tensor | None = None,
            chunk_size: int = 32,
    ) -> Tensor:
        """
        y_gt:   (B, T, C, H, W)
        u_traj: (B, T, u_dim), required if control_inputs=True
        returns: (B, T, z_dim)
        """
        assert y_gt.dim() == 5, f"Expected (B,T,C,H,W), got {tuple(y_gt.shape)}"

        b_size, n_steps, C, H, W = y_gt.shape
        y_gt_flat = y_gt.reshape(b_size * n_steps, C, H, W).contiguous()

        if self.control_inputs:
            if u_traj is None:
                raise ValueError("u_traj is required when control_inputs=True")

            assert u_traj.dim() == 3, f"Expected (B,T,u_dim), got {tuple(u_traj.shape)}"
            assert u_traj.shape[0] == b_size and u_traj.shape[1] == n_steps, (
                f"Incompatible shapes: y_gt={tuple(y_gt.shape)}, "
                f"u_traj={tuple(u_traj.shape)}"
            )

            u_dim = u_traj.shape[-1]
            u_flat = u_traj.reshape(b_size * n_steps, u_dim).contiguous()
        else:
            u_flat = None

        z_chunks = []
        for i in range(0, y_gt_flat.size(0), chunk_size):
            y_chunk = y_gt_flat[i:i + chunk_size]

            if self.control_inputs:
                u_chunk = u_flat[i:i + chunk_size]
                z_chunks.append(self.project(y_chunk, u_chunk))
            else:
                z_chunks.append(self.project(y_chunk, None))

        z_proj_flat = torch.cat(z_chunks, dim=0)
        return z_proj_flat.reshape(b_size, n_steps, self.z_dim)