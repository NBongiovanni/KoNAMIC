import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from KoNAMIC.core.models import VisionKoopModel


class ZTrajGen:
    def __init__(
        self,
        num_steps: int,
        koop_model: VisionKoopModel,
        drone_dim: int,
        delay: int,
        u_dim: int = 4,
    ):
        self.num_steps = num_steps
        self.koop_model = koop_model
        self.drone_dim = drone_dim
        self.delay = delay              # delay du modèle
        self.window_len = delay + 1     # nombre réel d'instants à empiler
        self.u_dim = u_dim

    @property
    def device(self):
        return next(self.koop_model.parameters()).device

    def pipeline(self, im_traj: Tensor) -> np.ndarray:
        """
        im_traj:
            2D -> [T, 1, H, W]
            3D -> [T, 2, H, W]
        """
        z_traj = []
        im_traj = im_traj.to(self.device)
        u_k = torch.zeros(1, self.u_dim, dtype=torch.float32, device=self.device)

        for k in tqdm(range(self.num_steps)):
            start = max(0, k - self.window_len + 1)
            window = im_traj[start:k + 1]

            if window.shape[0] < self.window_len:
                pad_len = self.window_len - window.shape[0]
                pad = im_traj[0].unsqueeze(0).repeat(pad_len, 1, 1, 1)
                window = torch.cat([pad, window], dim=0)

            z_k = self.encode_window(window, u_k)
            z_traj.append(z_k)

        return np.stack(z_traj)

    def encode_window(self, im_window: Tensor, u_k: Tensor) -> np.ndarray:
        """
        im_window: [window_len, C, H, W]
        y_k:       [1, window_len*C, H, W]
        """
        T, C, H, W = im_window.shape
        y_k = im_window.reshape(T * C, H, W).unsqueeze(0)

        with torch.no_grad():
            z_k = self.koop_model.project(y_k, u_k)

        print("model delay =", self.delay)
        print("window_len =", self.window_len)
        print("im_window.shape =", im_window.shape)
        print("y_k.shape =", y_k.shape)
        print("u_k.shape =", u_k.shape)
        print("expected per-view channels =", self.koop_model.auto_encoder.n_channels_encoder)

        return np.squeeze(z_k.detach().cpu().numpy())