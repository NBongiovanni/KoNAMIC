import numpy as np

from .traj_utils import traj_cosine, traj_ramp, traj_sine, traj_constant, traj_sine_cosine


class StateTrajGenerator:
    def __init__(self, drone_dim: int, num_steps: int, specs: dict, dt: float):
        self.dt = dt
        self.num_steps = num_steps
        self.specs = specs
        self.drone_dim = drone_dim
        self.x_ref_traj = None

    def pipeline(self) -> np.ndarray:
        x_ref_traj = self.generate_x_traj(self.num_steps, self.dt, self.specs)
        return x_ref_traj

    def generate_x_traj(self, num_steps: int, dt: float, specs: dict) -> np.ndarray:
        """
        Génère une trajectoire (n_steps, 6).
        specs[dim] doit contenir tous les paramètres requis.
        Exemple:
            {0: {"type": "ramp", "start":0.0, "end":1.0, "t_start":0.0, "t_end":2.0}}
        """
        if self.drone_dim == 2:
            x_ref_traj = np.zeros((num_steps, 6))
        elif self.drone_dim == 3:
            x_ref_traj = np.zeros((num_steps, 12))
        else:
            raise ValueError(f"Drone dimension inconnue: {self.drone_dim}")

        for dim, cfg in specs.items():
            kind = cfg["type"]

            if kind == "constant":
                x_ref_traj[:, dim] = traj_constant(num_steps, cfg["value"])

            elif kind == "ramp":
                x_ref_traj[:, dim] = traj_ramp(
                    num_steps,
                    dt,
                    cfg["start"],
                    cfg["end"],
                    cfg["t_start"],
                    cfg["t_end"]
                )

            elif kind == "sine":
                x_ref_traj[:, dim] = traj_sine(
                    num_steps,
                    dt,
                    cfg["amplitude"],
                    cfg["freq_hz"],
                    cfg["phase"]
                )

            elif kind == "cosine":
                x_ref_traj[:, dim] = traj_cosine(
                    num_steps,
                    dt,
                    cfg["amplitude"],
                    cfg["freq_hz"],
                    cfg["phase"]
                )

            elif kind == "sine_cosine":
                x_ref_traj[:, dim] = traj_sine_cosine(
                    num_steps,
                    dt,
                    cfg["amplitude"],
                    cfg["freq_hz"],
                    cfg["phase"]
                )
            else:
                raise ValueError(f"Type de trajectoire inconnu: {kind}")
        return x_ref_traj