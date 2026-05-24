from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from KoNAMIC.core.utils import build_relative_dataset_path, find_project_root
from .dataset_params import DatasetParams

class ImageProcessorMmap:
    def __init__(self, params: DatasetParams, phase: str):
        super().__init__()
        self.params = params
        self.phase = phase
        self.dataset_states = None
        self.dataset_inputs = None
        self.dataset_idx = None
        self.traj_len = self.params.train["num_steps_loaded"]

    def pipeline(self, num_simulations: int, im_size: int, num_steps_pred: int) -> None:
        root_dir = find_project_root()
        save_dir = root_dir / self._define_save_path()
        save_dir.mkdir(parents=True, exist_ok=True)
        traj_len = self.traj_len
        H = W = im_size

        # Vues selon la dimension
        if self.params.drone_dim == 3:
            views = ["left", "right"]
        else:
            views = [None]  # mono-vue
        V = len(views)
        print(f"[{self.phase}] Conversion of the PNG files in a unique numpy array")

        # 1) memmap pour les vision
        im_path = save_dir / "im_dataset_memmap.dat"
        im_dataset = np.memmap(
            im_path,
            dtype=np.uint8,
            mode="w+",
            shape=(num_simulations, traj_len, V, H, W),
        )
        name_png_dir = self._define_png_dir()

        def worker_load(args):
            i, j, v_idx, path = args
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Image not found: {path}")
            img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            return i, j, v_idx, img_resized

        paths = []
        for i in range(num_simulations):
            for j in range(traj_len):
                for v_idx, v in enumerate(views):
                    if v is None:
                        p = name_png_dir / f"traj_{i}" / f"step_{j}.png"
                    else:
                        p = name_png_dir / f"traj_{i}" / f"{v}" / f"step_{j}.png"
                    paths.append((i, j, v_idx, p))

        nb_threads = min(8, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=nb_threads) as executor:
            futures = {executor.submit(worker_load, tup): tup[:3] for tup in paths}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Chargement"):
                i, j, v_idx = futures[future]
                try:
                    _, _, _, img_proc = future.result()
                    im_dataset[i, j, v_idx, :, :] = img_proc
                except Exception as exc:
                    print(f"Erreur traitement image traj={i}, step={j}, view={v_idx} : {exc}")

        im_dataset.flush()
        del im_dataset

        # 3) Reshape + construction des paires (t, t+1)
        self.reorganise_dataset_mmap(
            im_path,
            num_simulations,
            traj_len,
            num_steps_pred,
            im_size,
            save_dir,
            num_views=V,
            delay=self.params.delay,
        )

    def _define_png_dir(self) -> Path:
        root_dir = find_project_root()
        version_dir = str(self.params.dataset_version)
        dim_dir = f"{self.params.drone_dim}d"
        return root_dir / "data" / "raw_images" / dim_dir / version_dir / self.phase

    def _define_save_path(self) -> Path:
        return build_relative_dataset_path(
            "vision",
            self.params.drone_dim,
            str(self.params.dataset_version),
            self.phase
        )

    def reorganise_dataset_mmap(
            self,
            im_path: Path,
            num_simulations: int,
            traj_len: int,
            seq_len: int,
            resolution: int,
            save_dir: Path,
            num_views: int = 1,
            delay: int = 1,
    ) -> None:
        if delay < 1:
            raise ValueError(f"delay must be >= 1, got {delay}")
        if traj_len <= delay:
            raise ValueError(f"traj_len={traj_len} must be > delay={delay} to build windows")

        print("Dataset reshape (memmap / par trajectoire)")
        total_samples, num_seq, H, W = self._compute_shapes(
            num_simulations, traj_len, seq_len, resolution, delay
        )
        im_dataset = np.memmap(
            im_path,
            dtype=np.uint8,
            mode="r",
            shape=(num_simulations, traj_len, num_views, H, W),
        )

        C = (delay + 1) * num_views
        y_sliced = np.memmap(
            save_dir / "dataset_memmap.dat",
            dtype=np.uint8,
            mode="w+",
            shape=(num_seq, seq_len, C, H, W),
        )
        y_flat = y_sliced.reshape(total_samples, C, H, W)

        pair_idx = 0
        for i in tqdm(range(num_simulations), desc="Reshape traj-by-traj"):
            traj = im_dataset[i]
            windows = self._traj_to_windows(traj, delay)
            n_i = windows.shape[0]
            y_flat[pair_idx:pair_idx + n_i] = windows
            pair_idx += n_i

        y_sliced.flush()
        self._write_metadata(
            save_dir,
            [num_seq, seq_len, C, H, W],
            num_views,
            delay
        )
        del y_sliced, y_flat, im_dataset
        os.remove(im_path)

    @staticmethod
    def _compute_shapes(
            num_simulations: int,
            traj_len: int,
            seq_len: int,
            resolution: int,
            delay: int,
    ) -> tuple[int, int, int, int]:

        n_samples_per_traj = traj_len - delay  # windows of length (delay+1)
        total_samples = num_simulations * n_samples_per_traj
        assert total_samples % seq_len == 0, "total_samples doit être divisible par seq_len"
        num_seq = total_samples // seq_len
        H = W = resolution
        return total_samples, num_seq, H, W

    @staticmethod
    def _traj_to_windows(traj: np.ndarray, delay: int) -> np.ndarray:
        """
        traj: (T, V, H, W)
        returns: (T-delay, (delay+1)*V, H, W)
        channel layout = concat([views@t-delay, ..., views@t])
        """
        T = traj.shape[0]
        # slices: traj[0:T-delay], traj[1:T-delay+1], ..., traj[delay:T]
        chunks = [traj[o: T - delay + o] for o in range(delay + 1)]
        return np.concatenate(chunks, axis=1)

    @staticmethod
    def _write_metadata(save_dir: Path, y_shape, n_views: int, delay: int) -> None:
        # e.g. delay=1 -> "concat([views@t-1, views@t])"
        #      delay=2 -> "concat([views@t-2, views@t-1, views@t])"
        parts = [f"views@t-{k}" for k in range(delay, 0, -1)] + ["views@t"]
        channel_layout = "concat([" + ", ".join(parts) + "])"

        metadata = {
            "y_shape": list(y_shape),
            "dtype": "uint8",
            "n_views": int(n_views),
            "delay": int(delay),
            "channel_layout": channel_layout,
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
