from tqdm import tqdm
import multiprocessing
from pathlib import Path

from KoNAMIC.core.utils import find_project_root
from .quad_drawer.quad_drawer_2d import QuadDrawer2D
from .quad_drawer.quad_drawer_3d_n_views import QuadDrawer3DNViews
from .dataset_params import DatasetParams

class ImDatasetGenerator:
    def __init__(
            self,
            params: DatasetParams,
            phase: str,
            raw_data: dict,
            num_simulations: int
    ):
        self.params = params
        self.phase = phase
        self.dataset_states = raw_data["x"]
        self.dataset_inputs = raw_data["u"]
        self.num_steps = self.params.train["num_steps_loaded"]
        self.num_trajs = num_simulations

        if params.drone_dim == 2:
            self.drawer = QuadDrawer2D(params.im_size, 128)
        elif params.drone_dim == 3:
            self.drawer = QuadDrawer3DNViews(params.im_size, thickness=1)
        else:
            raise ValueError(f"Unknown drone dimension: {params.drone_dim}")

    def generate_raw_images(self) -> None:
        root_dir = find_project_root()
        version_dir = str(self.params.dataset_version)
        dim_dir = f"{self.params.drone_dim}d"
        save_dir = root_dir / "data" / "raw_images" / dim_dir / version_dir
        print(self.phase + " dataset: raw image generation started")
        q = self.num_trajs // 4
        intervals = [
            (0,    q),
            (q,    2*q),
            (2*q,  3*q),
            (3 * q, self.num_trajs)
        ]

        processes = []
        for (start_i, end_i) in intervals:
            # On crée un process “ciblant” la méthode d’instance
            p = multiprocessing.Process(
                target=self._generate_raw_images_chunk,
                args=(start_i, end_i, save_dir)
            )
            p.start()
            processes.append(p)

        # On attend que tous les processus soient terminés
        for p in processes:
            p.join()
        print(f"[{self.phase}] dataset: raw image generation terminé.")

    def _generate_raw_images_chunk(
            self,
            traj_start_idx: int,
            traj_end_idx: int,
            save_dir: Path
        ) -> None:
        save_dir = save_dir / self.phase
        for i in tqdm(range(traj_start_idx, traj_end_idx)):
            self.generate_raw_traj(self.num_steps, i, save_dir)

    def generate_raw_traj(self, num_steps: int, traj_idx: int, save_dir: Path) -> None:
        x_traj = self.dataset_states[traj_idx]
        traj_dir = save_dir / f"traj_{traj_idx}"
        traj_dir.mkdir(parents=True, exist_ok=True)
        for j in range(num_steps):
            self.drawer.render_and_save(x_traj[j], save_dir, traj_idx, j)
