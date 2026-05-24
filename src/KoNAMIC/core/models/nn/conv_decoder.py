from typing import List, Tuple
from torch import Tensor
from torch import nn

from .layers import make_gn


class ConvDecoder(nn.Module):
    def __init__(
            self,
            out_channels: int,
            feature_shape: Tuple[int, int, int],
            out_planes: List[int],
            strides: List[int],
            eps: float = 1e-2,
            min_channels_per_group: int = 4,
            max_groups: int = 32,
    ):
        super().__init__()
        assert len(out_planes) == len(strides), (
            "out_planes et strides doivent avoir la même longueur"
        )

        self.act = nn.ReLU(inplace=True)
        self.eps = eps
        self.min_channels_per_group = min_channels_per_group
        self.max_groups = max_groups

        self.c_o, self.h_o, self.w_o = feature_shape
        self.decoder_convs = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()

        # ---- Decoder ----
        for i in range(len(out_planes) - 1):
            self.decoder_convs.append(
                nn.ConvTranspose2d(
                    out_planes[i],
                    out_planes[i + 1],
                    kernel_size=2,
                    stride=strides[i],
                    bias=False,
                )
            )
            self.decoder_norms.append(self.make_norm(out_planes[i + 1]))

        # Dernière couche de reconstruction: pas de normalisation, pas d'activation
        self.decoder_convs.append(
            nn.ConvTranspose2d(
                out_planes[-1],
                out_channels,
                kernel_size=2,
                stride=strides[-1],
                bias=True,
            )
        )
        self.decoder_norms.append(nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, D_latent) avec D_latent = c_o * h_o * w_o
        return : (B, out_channels, H, W)
        """
        out = x.view(-1, self.c_o, self.h_o, self.w_o)
        for conv, norm in zip(self.decoder_convs[:-1], self.decoder_norms[:-1]):
            out = self.act(norm(conv(out)))
        return self.decoder_convs[-1](out)

    def make_norm(self, num_channels: int) -> nn.Module:
        return make_gn(
            num_channels,
            self.max_groups,
            self.min_channels_per_group,
            self.eps
        )
