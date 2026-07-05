import torch
from torch import Tensor, nn

from .conv_encoder import ConvEncoder
from .conv_decoder import ConvDecoder
from .mlp_blocks import build_mlp
from .multi_view_encoder import MultiViewSharedEncoder
from .multi_view_decoder import MultiViewSharedDecoder


class MultiViewAutoEncoder(nn.Module):
    def __init__(self, params: dict, z_dim: int):
        super().__init__()
        self.params = params
        self.z_dim = z_dim
        delay = self.params["delay"]
        self.n_channels_encoder = delay + 1
        assert delay in [1, 2], f"Unsupported delay: {delay}"

        # ---- 1) Single-view CNN encoder (3 channels = t-2, t, t+1 for ONE view) ----
        self.single_view_encoder = ConvEncoder(
            num_in_channels=self.n_channels_encoder,
            H=self.params["H"],
            W=self.params["W"],
            out_planes=self.params["CNN"]["encoder"]["out_planes"],
            strides=self.params["CNN"]["encoder"]["strides"],
            n_split_layers=0,
        )

        # ---- 2) Multi-view wrapper (weight sharing) ----
        self.encoder_cnn = MultiViewSharedEncoder(
            single_view_encoder=self.single_view_encoder,
            split_indices=self.params["split_indices"],  # left=(0,2), right=(1,3)
        )

        w_dim_view = self.single_view_encoder.latent_dim
        self.w_dim = 2 * w_dim_view

        # ---- 3) MLPs unchanged, just use new w_dim ----
        self.encoder_mlp = build_mlp(
            self.w_dim+4,
            self.z_dim,
            self.params["MLP"]["hidden_dims"],
            self.params["activation"],
            layer_norm=self.params["MLP"].get("layer_norm", False),
            dropout=self.params["MLP"].get("dropout", 0.0),
        )

        self.decoder_mlp = build_mlp(
            self.z_dim,
            self.w_dim,
            self.params["MLP"]["hidden_dims"],
            self.params["activation"],
            layer_norm=self.params["MLP"].get("layer_norm", False),
            dropout=self.params["MLP"].get("dropout", 0.0),
        )

        # ---- 4) Decoder CNN ----
        # IMPORTANT: decoder expects feature_shape matching w_dim per sample.
        # We need a decoder that starts from (2 * c_o, h_o, w_o) OR decode each view separately.
        #
        # Recommended: decode each view separately with the same decoder (weight sharing), then concat.
        self.single_view_decoder = ConvDecoder(
            out_channels=self.params["CNN"]["decoder"]["num_out_channels"] // 2,  # e.g. 1 per view at t
            feature_shape=self.single_view_encoder.feature_shape,
            out_planes=self.params["CNN"]["decoder"]["out_planes"],
            strides=self.params["CNN"]["decoder"]["strides"],
        )
        self.decoder_cnn = MultiViewSharedDecoder(
            single_view_decoder=self.single_view_decoder,
            feature_shape=self.single_view_encoder.feature_shape,
        )

    def project(self, y: Tensor, u: Tensor) -> Tensor:
        w = self.encoder_cnn(y)

        wu = torch.cat([w, u], dim=1)
        return self.encoder_mlp(wu)

    def reconstruct(self, z: Tensor) -> Tensor:
        w = self.decoder_mlp(z)          # (B, 2*w_dim_view)
        return self.decoder_cnn(w)       # (B, 2, H, W) for left_t and right_t (example)

    def forward(self, y: Tensor, u:Tensor) -> Tensor:
        return self.reconstruct(self.project(y, u))

    def batch_projection(self, y_gt: Tensor, u_traj: Tensor, chunk_size: int = 32) -> Tensor:
        """
        y_gt:   (B, T, C, H, W)
        u_traj: (B, T, u_dim)
        returns: (B, T, z_dim)
        """
        assert y_gt.dim() == 5, f"Expected (B,T,C,H,W), got {tuple(y_gt.shape)}"
        assert u_traj.dim() == 3, f"Expected (B,T,u_dim), got {tuple(u_traj.shape)}"

        b_size, n_steps, C, H, W = y_gt.shape
        b_u, t_u, u_dim = u_traj.shape

        assert b_u == b_size and t_u == n_steps, (
            f"Incompatible shapes: y_gt={tuple(y_gt.shape)}, u_traj={tuple(u_traj.shape)}"
        )

        y_gt_flat = y_gt.reshape(b_size * n_steps, C, H, W).contiguous()
        u_traj_flat = u_traj.reshape(b_size * n_steps, u_dim).contiguous()

        z_chunks = []
        for i in range(0, y_gt_flat.size(0), chunk_size):
            y_chunk = y_gt_flat[i:i + chunk_size]
            u_chunk = u_traj_flat[i:i + chunk_size]
            z_chunks.append(self.project(y_chunk, u_chunk))

        z_proj_flat = torch.cat(z_chunks, dim=0)
        return z_proj_flat.reshape(b_size, n_steps, self.z_dim)