from typing import List, Tuple, Optional
import torch
from torch import Tensor, nn
from .layers import make_gn


class ConvEncoder(nn.Module):
    def __init__(
        self,
        num_in_channels: int,
        H: int,
        W: int,
        out_planes: List[int],
        strides: List[int],
        n_split_layers: int = 4,
        split_indices: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = ((0, 2), (1, 3)),
        eps: float = 1e-2,
        min_channels_per_group: int = 4,
        max_groups: int = 32,
    ):
        super().__init__()
        assert len(out_planes) == len(strides)
        assert n_split_layers >= 0
        assert n_split_layers <= len(out_planes)

        self.flatten = nn.Flatten(start_dim=1)
        self.act = nn.ReLU(inplace=True)
        self.eps = eps
        self.min_channels_per_group = min_channels_per_group
        self.max_groups = max_groups
        self.n_split_layers = n_split_layers
        self.split_indices = split_indices

        # ---------- cas sans split ----------
        if self.n_split_layers == 0:
            self.encoder_convs = nn.ModuleList()
            self.encoder_norms = nn.ModuleList()
            channels = [num_in_channels] + out_planes
            for i in range(len(out_planes)):
                layer = nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=3,
                    stride=strides[i],
                    padding=1,
                    bias=False
                )
                self.encoder_convs.append(layer)
                self.encoder_norms.append(self.make_norm(channels[i + 1]))

        # ---------- cas split ----------
        else:
            if split_indices is None: # vérifs split_indices
                raise ValueError("split_indices must be provided when n_split_layers>0")
            all_idx = split_indices[0] + split_indices[1]
            assert num_in_channels >= max(all_idx) + 1
            assert len(set(all_idx)) == 4

            # toutes les couches splittées doivent avoir out_planes[i] pair
            for i in range(self.n_split_layers):
                assert out_planes[i] % 2 == 0, (
                    f"out_planes[{i}]={out_planes[i]} must be even when split (need half/half)."
                )

            # --- couche 0 : garder EXACTEMENT les noms actuels si n_split_layers==1 ---
            c0_half = out_planes[0] // 2
            self.first_conv_left = nn.Conv2d(
                2,
                c0_half,
                3,
                stride=strides[0],
                padding=1,
                bias=False
            )
            self.first_conv_right = nn.Conv2d(
                2,
                c0_half,
                3,
                stride=strides[0],
                padding=1,
                bias=False
            )
            self.first_norm_left = self.make_norm(c0_half)
            self.first_norm_right = self.make_norm(c0_half)

            # --- couches splittées supplémentaires (si n_split_layers>1) ---
            self.split_convs_left = nn.ModuleList()
            self.split_convs_right = nn.ModuleList()
            self.split_norms_left = nn.ModuleList()
            self.split_norms_right = nn.ModuleList()

            for i in range(1, self.n_split_layers):
                c_in_half = out_planes[i - 1] // 2
                c_out_half = out_planes[i] // 2
                self.split_convs_left.append(nn.Conv2d(
                    c_in_half,
                    c_out_half,
                    3,
                    stride=strides[i],
                    padding=1,
                    bias=False
                ))
                self.split_convs_right.append(nn.Conv2d(
                    c_in_half,
                    c_out_half,
                    3,
                    stride=strides[i],
                    padding=1,
                    bias=False
                ))
                self.split_norms_left.append(self.make_norm(c_out_half))
                self.split_norms_right.append(self.make_norm(c_out_half))

            # --- shared à partir de i=n_split_layers ---
            self.shared_convs = nn.ModuleList()
            self.shared_norms = nn.ModuleList()

            for i in range(self.n_split_layers, len(out_planes)):
                c_in = out_planes[i - 1]  # après concat, on est à out_planes[i-1]
                c_out = out_planes[i]
                self.shared_convs.append(nn.Conv2d(
                    c_in,
                    c_out,
                    3,
                    stride=strides[i],
                    padding=1,
                    bias=False
                ))
                self.shared_norms.append(self.make_norm(c_out))

        # --- feature_shape via dummy forward (comme avant) ---
        with torch.no_grad():
            dummy = torch.zeros(1, num_in_channels, H, W)
            out = self._forward_convs(dummy)
            _, self.c_o, self.h_o, self.w_o = out.shape
            self.latent_dim = self.c_o * self.h_o * self.w_o

    def _forward_convs(self, x: Tensor) -> Tensor:
        out = x

        if self.n_split_layers == 0:
            for conv, norm in zip(self.encoder_convs, self.encoder_norms):
                out = self.act(norm(conv(out)))
            return out

        (l0, l1), (r0, r1) = self.split_indices  # type: ignore[misc]
        xL = out[:, [l0, l1], :, :]
        xR = out[:, [r0, r1], :, :]

        # couche 0
        hL = self.act(self.first_norm_left(self.first_conv_left(xL)))
        hR = self.act(self.first_norm_right(self.first_conv_right(xR)))

        # couches 1..n_split_layers-1
        for convL, normL, convR, normR in zip(
            self.split_convs_left, self.split_norms_left,
            self.split_convs_right, self.split_norms_right
        ):
            hL = self.act(normL(convL(hL)))
            hR = self.act(normR(convR(hR)))

        # concat puis shared
        out = torch.cat([hL, hR], dim=1)

        for conv, norm in zip(self.shared_convs, self.shared_norms):
            out = self.act(norm(conv(out)))
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self.flatten(self._forward_convs(x))

    def make_norm(self, num_channels: int) -> nn.Module:
        return make_gn(
            num_channels,
            self.max_groups,
            self.min_channels_per_group,
            self.eps
        )

    @property
    def feature_shape(self) -> Tuple[int, int, int]:
        return self.c_o, self.h_o, self.w_o